use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/tru_cuda.cu");
    println!("cargo:rerun-if-changed=src/tru_cuda.h");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_windows = target_os == "windows";

    let nvcc_candidates = [
        "/usr/local/cuda-13.4/bin/nvcc",
        "/usr/local/cuda-13/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "nvcc",
    ];
    let nvcc_path = nvcc_candidates
        .iter()
        .find(|&&p| std::path::Path::new(p).exists() || p == "nvcc")
        .unwrap_or(&"nvcc");

    let mut nvcc_cmd = Command::new(nvcc_path);
    nvcc_cmd.args(&[
        "-O3",
        "-std=c++17",
    ]);

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "aarch64" {
        nvcc_cmd.args(&[
            "-D__Float32x4_t=int",
            "-D__Float64x2_t=int",
            "-D__SVFloat32_t=int",
            "-D__SVFloat64_t=int",
            "-D__SVBool_t=int",
        ]);
    }

    if let Ok(arch_env) = env::var("CUDA_ARCH") {
        let arch = arch_env.trim();
        let virt = if arch.starts_with("sm_") {
            arch.replace("sm_", "compute_")
        } else if !arch.starts_with("compute_") {
            format!("compute_{}", arch)
        } else {
            arch.to_string()
        };
        let real = if arch.starts_with("compute_") {
            arch.replace("compute_", "sm_")
        } else if !arch.starts_with("sm_") {
            format!("sm_{}", arch)
        } else {
            arch.to_string()
        };
        nvcc_cmd.args(&["-gencode", &format!("arch={},code={}", virt, real)]);
        nvcc_cmd.args(&["-gencode", &format!("arch={},code={}", virt, virt)]);
    } else {
        // Target all supported Jetson embedded and desktop/server architectures:
        // - compute_53: Jetson Nano, Jetson TX1 (Maxwell)
        // - compute_62: Jetson TX2 (Pascal)
        // - compute_72: Jetson AGX Xavier, Xavier NX (Volta)
        // - compute_75: Turing (RTX 20xx, GTX 16xx)
        // - compute_80: Ampere Datacenter (A100)
        // - compute_86: Ampere Desktop (RTX 30xx)
        // - compute_87: Jetson AGX Orin, Orin NX, Orin Nano (Ampere)
        // - compute_89: Ada Lovelace (RTX 40xx)
        // - compute_90: Hopper Datacenter (H100)
        // - compute_100: Blackwell Datacenter (B100/B200 / Jetson Thor)
        // - compute_120: Blackwell Desktop (RTX 50xx)
        let candidate_arches = [
            ("compute_53", "sm_53"),
            ("compute_62", "sm_62"),
            ("compute_72", "sm_72"),
            ("compute_75", "sm_75"),
            ("compute_80", "sm_80"),
            ("compute_86", "sm_86"),
            ("compute_87", "sm_87"),
            ("compute_89", "sm_89"),
            ("compute_90", "sm_90"),
            ("compute_100", "sm_100"),
            ("compute_120", "sm_120"),
        ];

        let help_output = Command::new(nvcc_path)
            .arg("--help")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();

        let mut highest_virtual: Option<&str> = None;
        for (virt, real) in &candidate_arches {
            if help_output.is_empty() || help_output.contains(virt) {
                nvcc_cmd.args(&["-gencode", &format!("arch={},code={}", virt, real)]);
                highest_virtual = Some(virt);
            }
        }

        if let Some(highest) = highest_virtual {
            nvcc_cmd.args(&["-gencode", &format!("arch={},code={}", highest, highest)]);
        }
    }

    if is_windows {
        let lib_path = out_dir.join("tru_cuda.lib");
        nvcc_cmd.args(&[
            "-lib",
            "src/tru_cuda.cu",
            "-o", lib_path.to_str().unwrap(),
        ]);
        let status = nvcc_cmd.status().expect("Failed to execute nvcc");
        if !status.success() {
            panic!("nvcc compilation failed");
        }

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=tru_cuda");

        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={}\\lib\\x64", cuda_path);
        }
        println!("cargo:rustc-link-lib=cudart");
    } else {
        let obj_path = out_dir.join("tru_cuda.o");
        let lib_path = out_dir.join("libtru_cuda.a");

        nvcc_cmd.args(&[
            "--compiler-options", "-fPIC",
            "-c", "src/tru_cuda.cu",
            "-o", obj_path.to_str().unwrap(),
        ]);
        let status = nvcc_cmd.status().expect("Failed to execute nvcc");
        if !status.success() {
            panic!("nvcc compilation failed");
        }

        let status = Command::new("ar")
            .args(&[
                "crs",
                lib_path.to_str().unwrap(),
                obj_path.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute ar");

        if !status.success() {
            panic!("ar archive failed");
        }

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        let cuda_lib_dirs = [
            "/usr/local/cuda-13.4/targets/x86_64-linux/lib",
            "/usr/local/cuda-13.4/lib64",
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/local/cuda/targets/aarch64-linux/lib",
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/lib",
            "/usr/lib/aarch64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu",
        ];
        for dir in &cuda_lib_dirs {
            if std::path::Path::new(dir).exists() {
                println!("cargo:rustc-link-search=native={}", dir);
            }
        }
        println!("cargo:rustc-link-lib=static=tru_cuda");
        println!("cargo:rustc-link-lib=static=cudart_static");
        println!("cargo:rustc-link-lib=rt");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=stdc++");
    }
}
