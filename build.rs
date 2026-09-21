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
        "-gencode", "arch=compute_75,code=sm_75",
        "-gencode", "arch=compute_80,code=sm_80",
        "-gencode", "arch=compute_86,code=sm_86",
        "-gencode", "arch=compute_89,code=sm_89",
        "-gencode", "arch=compute_90,code=sm_90",
        "-gencode", "arch=compute_120,code=sm_120",
        "-gencode", "arch=compute_120,code=compute_120",
    ]);

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
        if std::path::Path::new("/usr/local/cuda-13.4/targets/x86_64-linux/lib").exists() {
            println!("cargo:rustc-link-search=native=/usr/local/cuda-13.4/targets/x86_64-linux/lib");
        }
        if std::path::Path::new("/usr/local/cuda-13.4/lib64").exists() {
            println!("cargo:rustc-link-search=native=/usr/local/cuda-13.4/lib64");
        } else if std::path::Path::new("/usr/local/cuda/lib64").exists() {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        }
        println!("cargo:rustc-link-lib=static=tru_cuda");
        println!("cargo:rustc-link-lib=static=cudart_static");
        println!("cargo:rustc-link-lib=rt");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=stdc++");
    }
}
