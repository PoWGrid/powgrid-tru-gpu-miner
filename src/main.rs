use serde_json::{json, Value};
mod opencl_backend;
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn get_time_str() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let secs = now % 60;
    let mins = (now / 60) % 60;
    let hours = (now / 3600) % 24;
    format!("{:02}:{:02}:{:02}", hours, mins, secs)
}

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(filename: *const std::ffi::c_char, flag: std::ffi::c_int) -> *mut std::ffi::c_void;
    fn dlsym(handle: *mut std::ffi::c_void, symbol: *const std::ffi::c_char) -> *mut std::ffi::c_void;
}

#[cfg(windows)]
unsafe extern "system" {
    fn LoadLibraryA(lpLibFileName: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn GetProcAddress(hModule: *mut std::ffi::c_void, lpProcName: *const std::ffi::c_char) -> *mut std::ffi::c_void;
}

#[inline]
unsafe fn load_lib(name: *const std::ffi::c_char) -> *mut std::ffi::c_void {
    #[cfg(unix)]
    return dlopen(name, 1);
    #[cfg(windows)]
    return LoadLibraryA(name);
}

#[inline]
unsafe fn load_sym(lib: *mut std::ffi::c_void, name: *const std::ffi::c_char) -> *mut std::ffi::c_void {
    #[cfg(unix)]
    return dlsym(lib, name);
    #[cfg(windows)]
    return GetProcAddress(lib, name);
}

pub struct GpuTelemetry {
    pub temp_c: Option<u32>,
    pub power_w: Option<f64>,
    pub efficiency_mhw: Option<f64>,
}

pub struct NvmlMonitor {
    pub device: *mut std::ffi::c_void,
    pub fn_temp: Option<unsafe extern "C" fn(*mut std::ffi::c_void, u32, *mut u32) -> u32>,
    pub fn_power: Option<unsafe extern "C" fn(*mut std::ffi::c_void, *mut u32) -> u32>,
    pub fn_set_power: Option<unsafe extern "C" fn(*mut std::ffi::c_void, u32) -> u32>,
}

unsafe impl Send for NvmlMonitor {}
unsafe impl Sync for NvmlMonitor {}

impl NvmlMonitor {
    pub fn new(device_id: u32) -> Self {
        unsafe {
            #[cfg(unix)]
            let lib_name = b"libnvidia-ml.so.1\0";
            #[cfg(windows)]
            let lib_name = b"nvml.dll\0";
            let lib = load_lib(lib_name.as_ptr() as *const std::ffi::c_char);
            if lib.is_null() {
                return Self {
                    device: std::ptr::null_mut(),
                    fn_temp: None,
                    fn_power: None,
                    fn_set_power: None,
                };
            }

            let init_sym = load_sym(lib, b"nvmlInit_v2\0".as_ptr() as *const std::ffi::c_char);
            let get_handle_sym = load_sym(lib, b"nvmlDeviceGetHandleByIndex_v2\0".as_ptr() as *const std::ffi::c_char);
            let get_temp_sym = load_sym(lib, b"nvmlDeviceGetTemperature\0".as_ptr() as *const std::ffi::c_char);
            let get_power_sym = load_sym(lib, b"nvmlDeviceGetPowerUsage\0".as_ptr() as *const std::ffi::c_char);
            let set_power_sym = load_sym(lib, b"nvmlDeviceSetPowerManagementLimit\0".as_ptr() as *const std::ffi::c_char);

            if init_sym.is_null() || get_handle_sym.is_null() {
                return Self {
                    device: std::ptr::null_mut(),
                    fn_temp: None,
                    fn_power: None,
                    fn_set_power: None,
                };
            }

            let init_fn: unsafe extern "C" fn() -> u32 = std::mem::transmute(init_sym);
            let get_handle_fn: unsafe extern "C" fn(u32, *mut *mut std::ffi::c_void) -> u32 =
                std::mem::transmute(get_handle_sym);

            if init_fn() != 0 {
                return Self {
                    device: std::ptr::null_mut(),
                    fn_temp: None,
                    fn_power: None,
                    fn_set_power: None,
                };
            }

            let mut dev: *mut std::ffi::c_void = std::ptr::null_mut();
            if get_handle_fn(device_id, &mut dev) != 0 || dev.is_null() {
                return Self {
                    device: std::ptr::null_mut(),
                    fn_temp: None,
                    fn_power: None,
                    fn_set_power: None,
                };
            }

            let fn_temp = if !get_temp_sym.is_null() {
                Some(std::mem::transmute(get_temp_sym))
            } else {
                None
            };

            let fn_power = if !get_power_sym.is_null() {
                Some(std::mem::transmute(get_power_sym))
            } else {
                None
            };

            let fn_set_power = if !set_power_sym.is_null() {
                Some(std::mem::transmute(set_power_sym))
            } else {
                None
            };

            Self {
                device: dev,
                fn_temp,
                fn_power,
                fn_set_power,
            }
        }
    }

    pub fn apply_power_limit(&self, device_id: u32, watts: u32) -> bool {
        let mut success = false;
        if let Some(f) = self.fn_set_power {
            if !self.device.is_null() {
                let res = unsafe { f(self.device, watts * 1000) };
                if res == 0 {
                    success = true;
                }
            }
        }
        if !success {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(&["-i", &device_id.to_string(), "-pl", &watts.to_string()])
                .output()
            {
                if output.status.success() {
                    success = true;
                }
            }
        }
        success
    }

    pub fn sample(&self, current_mhs: f64) -> GpuTelemetry {
        if self.device.is_null() {
            return GpuTelemetry {
                temp_c: None,
                power_w: None,
                efficiency_mhw: None,
            };
        }

        let mut temp_c = None;
        if let Some(f) = self.fn_temp {
            let mut t = 0u32;
            if unsafe { f(self.device, 0, &mut t) } == 0 {
                temp_c = Some(t);
            }
        }

        let mut power_w = None;
        if let Some(f) = self.fn_power {
            let mut p = 0u32;
            if unsafe { f(self.device, &mut p) } == 0 {
                power_w = Some(p as f64 / 1000.0);
            }
        }

        let efficiency_mhw = match (power_w, current_mhs) {
            (Some(w), m) if w > 5.0 && m > 0.0 => Some(m / w),
            _ => None,
        };

        GpuTelemetry {
            temp_c,
            power_w,
            efficiency_mhw,
        }
    }
}

pub struct GpuDeviceTelemetry {
    pub device_id: i32,
    pub name: String,
    pub arch_name: String,
    pub sm_count: i32,
    pub batch_size: u32,
    pub nvml: NvmlMonitor,
    pub current_mhs: f64,
    pub total_hashes: u64,
    pub last_tick_hashes: u64,
}

pub struct Dashboard {
    pub is_tty: bool,
    pub start_time: Instant,
    pub last_ui_update: Instant,
    pub last_log_time: Instant,
    pub gpus: Vec<GpuDeviceTelemetry>,
    pub preset: String,
    pub pool_url: String,
    pub active_wallet: String,
    pub active_worker: String,
    pub is_pool_mode: bool,
    pub accepted_shares: u64,
    pub rejected_shares: u64,
    pub worker_diff: f64,
    pub events: VecDeque<String>,
    pub power_cap: Option<u32>,
}

use unicode_width::UnicodeWidthChar;

fn visible_len(s: &str) -> usize {
    let mut len = 0;
    let mut in_esc = false;
    for c in s.chars() {
        if c == '\x1b' {
            in_esc = true;
        } else if in_esc {
            if c == 'm' {
                in_esc = false;
            }
        } else {
            len += c.width().unwrap_or(0);
        }
    }
    len
}

fn truncate_visible(s: &str, max_width: usize) -> String {
    let mut out = String::new();
    let mut current_width = 0;
    let mut in_esc = false;
    for c in s.chars() {
        if c == '\x1b' {
            in_esc = true;
            out.push(c);
        } else if in_esc {
            out.push(c);
            if c == 'm' {
                in_esc = false;
            }
        } else {
            let cw = c.width().unwrap_or(0);
            if current_width + cw <= max_width {
                out.push(c);
                current_width += cw;
            } else {
                break;
            }
        }
    }
    out.push_str("\x1b[0m");
    out
}

fn box_row(content: &str, inner_width: usize) -> String {
    let vlen = visible_len(content);
    let padding = if vlen < inner_width { inner_width - vlen } else { 0 };
    format!("│ {}{} │\x1b[K\n", content, " ".repeat(padding))
}

fn two_col_row(left: &str, left_col_width: usize, right: &str, inner_width: usize) -> String {
    let l_vlen = visible_len(left);
    let (clean_left, l_vlen) = if l_vlen > left_col_width.saturating_sub(1) {
        let t = truncate_visible(left, left_col_width.saturating_sub(1));
        let len = visible_len(&t);
        (t, len)
    } else {
        (left.to_string(), l_vlen)
    };
    let l_pad = if l_vlen < left_col_width { left_col_width - l_vlen } else { 1 };
    let r_vlen = visible_len(right);
    let max_r = inner_width.saturating_sub(l_vlen + l_pad);
    let (clean_right, r_vlen) = if r_vlen > max_r {
        let t = truncate_visible(right, max_r);
        let len = visible_len(&t);
        (t, len)
    } else {
        (right.to_string(), r_vlen)
    };
    let total_used = l_vlen + l_pad + r_vlen;
    let r_pad = if total_used < inner_width { inner_width - total_used } else { 0 };
    format!("│ {}{}{}{} │\x1b[K\n", clean_left, " ".repeat(l_pad), clean_right, " ".repeat(r_pad))
}

fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut res = String::new();
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    for (i, &c) in chars.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            res.push(',');
        }
        res.push(c);
    }
    res
}

static RESIZE_EVENT: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn sigwinch_handler(_: libc::c_int) {
    RESIZE_EVENT.store(true, Ordering::Relaxed);
}

impl Dashboard {
    pub fn new(
        gpus: Vec<GpuDeviceTelemetry>,
        preset: String,
        pool_url: String,
        active_wallet: String,
        active_worker: String,
        is_pool_mode: bool,
        is_hiveos: bool,
    ) -> Self {
        let is_tty = !is_hiveos && std::io::stdout().is_terminal();
        if is_tty {
            #[cfg(unix)]
            unsafe {
                libc::signal(libc::SIGWINCH, sigwinch_handler as usize);
            }
            print!("\x1b[?1049h\x1b[?7l\x1b[2J\x1b[H\x1b[?25l");
            std::io::stdout().flush().ok();
        }
        let now = Instant::now();
        Self {
            is_tty,
            start_time: now,
            last_ui_update: now,
            last_log_time: now,
            gpus,
            preset,
            pool_url,
            active_wallet,
            active_worker,
            is_pool_mode,
            accepted_shares: 0,
            rejected_shares: 0,
            worker_diff: 1.0,
            events: VecDeque::with_capacity(8),
            power_cap: None,
        }
    }

    pub fn set_power_cap(&mut self, cap: u32) {
        self.power_cap = Some(cap);
    }

    pub fn add_event(&mut self, ev: String) {
        if !self.is_tty {
            println!("{}", ev);
        }
        if self.events.len() >= 5 {
            self.events.pop_front();
        }
        self.events.push_back(ev);
    }

    pub fn render(
        &mut self,
        current_height: u32,
        bits_hex: &str,
        blocks_found: u32,
        force: bool,
    ) {
        let now = Instant::now();
        let total_current_mhs: f64 = self.gpus.iter().map(|g| g.current_mhs).sum();
        let total_hashes: u64 = self.gpus.iter().map(|g| g.total_hashes).sum();

        if !self.is_tty {
            if force || now.duration_since(self.last_log_time) >= Duration::from_secs(5) {
                let uptime = self.start_time.elapsed().as_secs();
                let short_w = if self.active_wallet.len() >= 14 {
                    format!("{}...{}", &self.active_wallet[0..8], &self.active_wallet[self.active_wallet.len() - 4..])
                } else {
                    self.active_wallet.clone()
                };

                let gpu_summary: Vec<String> = self.gpus.iter().map(|g| {
                    let tele = g.nvml.sample(g.current_mhs);
                    match tele.temp_c {
                        Some(t) => format!("GPU{}: {:.1}MH/s ({}°C)", g.device_id, g.current_mhs, t),
                        None => format!("GPU{}: {:.1}MH/s", g.device_id, g.current_mhs),
                    }
                }).collect();

                let gpu_info_str = if self.gpus.len() > 1 {
                    format!(" [{}]", gpu_summary.join(", "))
                } else if !self.gpus.is_empty() {
                    let tele = self.gpus[0].nvml.sample(self.gpus[0].current_mhs);
                    match (tele.temp_c, tele.power_w) {
                        (Some(t), Some(w)) => format!(" | {}°C, {:.1}W", t, w),
                        (Some(t), None) => format!(" | {}°C", t),
                        _ => String::new(),
                    }
                } else {
                    String::new()
                };

                if self.is_pool_mode {
                    println!(
                        "[{:02}:{:02}:{:02}] ⛏️  Speed: {:.2} MH/s ({:.2} GH/s){} | Shares: {}/{} | Blocks: {} | Diff: {:.2} | Worker: {}",
                        uptime / 3600, (uptime % 3600) / 60, uptime % 60,
                        total_current_mhs, total_current_mhs / 1000.0,
                        gpu_info_str,
                        self.accepted_shares, self.rejected_shares,
                        blocks_found,
                        self.worker_diff,
                        self.active_worker
                    );
                } else {
                    println!(
                        "[{:02}:{:02}:{:02}] ⛏️  Speed: {:.2} MH/s ({:.2} GH/s){} | Total: {}M | Blocks: {} | Height: #{} | Wallet: {}",
                        uptime / 3600, (uptime % 3600) / 60, uptime % 60,
                        total_current_mhs, total_current_mhs / 1000.0,
                        gpu_info_str,
                        total_hashes / 1_000_000,
                        blocks_found,
                        current_height,
                        short_w
                    );
                }
                self.last_log_time = now;
            }
            return;
        }

        if !force && now.duration_since(self.last_ui_update) < Duration::from_millis(500) {
            return;
        }
        self.last_ui_update = now;

        let uptime_secs = self.start_time.elapsed().as_secs();
        let hours = uptime_secs / 3600;
        let mins = (uptime_secs % 3600) / 60;
        let secs = uptime_secs % 60;

        let total_elapsed = self.start_time.elapsed().as_secs_f64();
        let avg_mhs = if total_elapsed > 0.0 {
            (total_hashes as f64 / total_elapsed) / 1_000_000.0
        } else {
            0.0
        };

        let short_w = if self.active_wallet.len() >= 14 {
            format!("{}...{}", &self.active_wallet[0..10], &self.active_wallet[self.active_wallet.len() - 4..])
        } else {
            self.active_wallet.clone()
        };

        let cap_str = match self.power_cap {
            Some(cap) => format!(" / {}W", cap),
            None => String::new(),
        };

        let mut out = String::with_capacity(4096);
        if RESIZE_EVENT.swap(false, Ordering::Relaxed) {
            out.push_str("\x1b[2J\x1b[H");
        } else {
            out.push_str("\x1b[H"); // Zero-flicker cursor home
        }

        let border_top = format!("╭{}╮\x1b[K\n", "─".repeat(76));
        let border_div = format!("├{}┤\x1b[K\n", "─".repeat(76));
        let border_bot = format!("╰{}╯\x1b[K\n", "─".repeat(76));

        out.push_str(&border_top);
        if self.is_pool_mode {
            out.push_str(&box_row("\x1b[1;36m► POWGRID TRU ($TRU) HIGH-PERFORMANCE TRUHASH GPU MINER v1.3.2\x1b[0m", 74));
        } else {
            out.push_str(&box_row("\x1b[1;36m► TRU FAT MINER ($TRU) HIGH-SPEED SOLO GPU ENGINE v2.3\x1b[0m", 74));
        }

        let pool_display = if self.pool_url.len() > 30 {
            format!("{}...", &self.pool_url[..27])
        } else {
            self.pool_url.clone()
        };

        out.push_str(&two_col_row(
            &format!("\x1b[90mPool:\x1b[0m \x1b[1;37m{}\x1b[0m", pool_display),
            39,
            "\x1b[90mStatus:\x1b[0m \x1b[1;32m● ONLINE (Connected)\x1b[0m",
            74,
        ));

        let mode_label = if self.is_pool_mode { "PPLNS Pool" } else { "SOLO" };
        out.push_str(&two_col_row(
            &format!("\x1b[90mWallet:\x1b[0m \x1b[36m{}\x1b[0m", short_w),
            39,
            &format!("\x1b[90mWorker:\x1b[0m \x1b[1;33m{}\x1b[0m \x1b[90m({})\x1b[0m", self.active_worker, mode_label),
            74,
        ));

        out.push_str(&border_div);
        out.push_str(&box_row(&format!("\x1b[1;35mHARDWARE & GPU ENGINE CONFIGURATION ({} GPUs Active)\x1b[0m", self.gpus.len()), 74));

        if self.gpus.len() == 1 {
            let gpu = &self.gpus[0];
            let tele = gpu.nvml.sample(gpu.current_mhs);
            let gpu_trim = if let Some(pos) = gpu.name.to_lowercase().find("rtx") {
                let rest = &gpu.name[pos..];
                let clean = rest.trim_end_matches(" GPU").trim();
                if clean.len() > 16 { &clean[..16] } else { clean }
            } else if gpu.name.len() > 16 {
                &gpu.name[..16]
            } else {
                &gpu.name
            };
            let sm_unit = if gpu.arch_name.contains("OpenCL") || gpu.arch_name.contains("AMD") || gpu.arch_name.contains("Intel") {
                "CUs"
            } else {
                "SMs"
            };
            let sm_suffix = format!("({} {})", gpu.sm_count, sm_unit);
            let max_name = 24usize.saturating_sub(sm_suffix.len() + 1);
            let name_clean = if gpu_trim.len() > max_name { &gpu_trim[..max_name] } else { gpu_trim };
            let dev_label = format!("{} {}", name_clean, sm_suffix);

            let arch_clean = if gpu.arch_name.contains("Blackwell") {
                "Blackwell"
            } else if gpu.arch_name.contains("Ada") {
                "Ada"
            } else if gpu.arch_name.contains("Ampere") {
                "Ampere"
            } else if gpu.arch_name.contains("Hopper") {
                "Hopper"
            } else if gpu.arch_name.contains("Turing") {
                "Turing"
            } else if let Some(first) = gpu.arch_name.split('(').next() {
                let trimmed = first.trim();
                if trimmed.is_empty() { "CUDA" } else if trimmed.len() > 12 { &trimmed[..12] } else { trimmed }
            } else {
                "CUDA"
            };
            let algo_str = format!("TRUHash ({})", arch_clean);
            out.push_str(&two_col_row(
                &format!("\x1b[90mPrimary GPU:\x1b[0m \x1b[1;37m{}\x1b[0m", dev_label),
                39,
                &format!("\x1b[90mAlgorithm  :\x1b[0m \x1b[1;32m{}\x1b[0m", algo_str),
                74,
            ));

            let tele_line = match (tele.temp_c, tele.power_w, tele.efficiency_mhw) {
                (Some(t), Some(w), Some(eff)) => {
                    let temp_color = if t < 70 { "\x1b[1;32m" } else if t < 83 { "\x1b[1;33m" } else { "\x1b[1;31m" };
                    format!("{}{}\x1b[0m°C | {:.0}W{} | \x1b[1;36m{:.1}\x1b[0m MH/W", temp_color, t, w, cap_str, eff)
                }
                (Some(t), Some(w), None) => {
                    let temp_color = if t < 70 { "\x1b[1;32m" } else if t < 83 { "\x1b[1;33m" } else { "\x1b[1;31m" };
                    format!("{}{}\x1b[0m°C | {:.0}W{}", temp_color, t, w, cap_str)
                }
                (Some(t), None, _) => format!("{}°C", t),
                _ => "\x1b[90mN/A\x1b[0m".to_string(),
            };

            out.push_str(&two_col_row(
                &format!("\x1b[90mTelemetry  :\x1b[0m {}", tele_line),
                39,
                &format!("\x1b[90mBatch Size :\x1b[0m \x1b[1;33m{}M\x1b[0m ({})", gpu.batch_size / 1_000_000, self.preset),
                74,
            ));
        } else {
            for gpu in &self.gpus {
                let tele = gpu.nvml.sample(gpu.current_mhs);
                let gpu_short = if let Some(pos) = gpu.name.to_lowercase().find("rtx") {
                    let rest = &gpu.name[pos..];
                    let clean = rest.trim_end_matches(" GPU").trim();
                    if clean.len() > 14 { &clean[..14] } else { clean }
                } else if gpu.name.len() > 14 {
                    &gpu.name[..14]
                } else {
                    &gpu.name
                };
                let tele_str = match (tele.temp_c, tele.power_w) {
                    (Some(t), Some(w)) => format!("{}°C {:.0}W", t, w),
                    (Some(t), None) => format!("{}°C", t),
                    _ => "N/A".to_string(),
                };
                let left = format!("\x1b[1;37mGPU #{}:\x1b[0m \x1b[90m{}\x1b[0m ({})", gpu.device_id, gpu_short, gpu.arch_name);
                let right = format!("\x1b[1;32m{:>7.2} MH/s\x1b[0m | \x1b[90m{}\x1b[0m", gpu.current_mhs, tele_str);
                out.push_str(&two_col_row(&left, 39, &right, 74));
            }
        }

        out.push_str(&border_div);
        out.push_str(&box_row("\x1b[1;33mMINING TELEMETRY\x1b[0m", 74));

        out.push_str(&two_col_row(
            &format!("\x1b[90mHashrate(Now)  :\x1b[0m \x1b[1;32m{:>7.2} MH/s\x1b[0m", total_current_mhs),
            39,
            &format!("\x1b[90mUptime      :\x1b[0m \x1b[1;37m{:02}:{:02}:{:02}\x1b[0m", hours, mins, secs),
            74,
        ));

        let diff_str = if self.is_pool_mode {
            format!("{:.2} (Vardiff)", self.worker_diff)
        } else {
            format!("0x{}", bits_hex)
        };

        out.push_str(&two_col_row(
            &format!("\x1b[90mHashrate(Avg)  :\x1b[0m \x1b[1;32m{:>7.2} MH/s\x1b[0m", avg_mhs),
            39,
            &format!("\x1b[90mTarget Diff :\x1b[0m \x1b[1;36m{}\x1b[0m", diff_str),
            74,
        ));

        let total_shares = self.accepted_shares + self.rejected_shares;
        let acc_pct = if total_shares > 0 {
            (self.accepted_shares as f64 / total_shares as f64) * 100.0
        } else {
            100.0
        };
        let rej_pct = if total_shares > 0 {
            (self.rejected_shares as f64 / total_shares as f64) * 100.0
        } else {
            0.0
        };

        if self.is_pool_mode {
            let rej_color = if self.rejected_shares > 0 { "\x1b[1;31m" } else { "\x1b[1;32m" };
            out.push_str(&two_col_row(
                &format!("\x1b[90mShares (Acc)   :\x1b[0m \x1b[1;32m{}\x1b[0m ({:.1}%)", self.accepted_shares, acc_pct),
                39,
                &format!("\x1b[90mRejected    :\x1b[0m {}{}\x1b[0m ({:.1}%)", rej_color, self.rejected_shares, rej_pct),
                74,
            ));
        } else {
            out.push_str(&two_col_row(
                &format!("\x1b[90mTotal Hashes   :\x1b[0m \x1b[1;37m{}\x1b[0m", format_number(total_hashes)),
                39,
                &format!("\x1b[90mHeight      :\x1b[0m \x1b[1;33m#{}\x1b[0m", current_height),
                74,
            ));
        }

        let blocks_str = if blocks_found > 0 {
            format!("\x1b[1;93m★ {} BLOCK{}\x1b[0m", blocks_found, if blocks_found > 1 { "S" } else { "" })
        } else {
            "\x1b[1;37m★ 0\x1b[0m \x1b[90m(hunting)\x1b[0m".to_string()
        };

        out.push_str(&two_col_row(
            &format!("\x1b[90mBlocks Found   :\x1b[0m {}", blocks_str),
            39,
            &format!("\x1b[90mBlock Height:\x1b[0m \x1b[1;36m#{}\x1b[0m", current_height),
            74,
        ));

        out.push_str(&border_div);
        out.push_str(&box_row("\x1b[1;34mRECENT MINING & NETWORK EVENTS\x1b[0m", 74));

        let ev_len = self.events.len();
        for i in 0..5 {
            if i < ev_len {
                let ev = &self.events[ev_len - 1 - i];
                out.push_str(&box_row(&truncate_visible(ev, 70), 74));
            } else {
                out.push_str(&box_row("\x1b[90m--\x1b[0m", 74));
            }
        }

        out.push_str(&border_bot);
        out.push_str("  \x1b[90m[Mining active in continuous CUDA dual-stream | CTRL+C to exit]\x1b[0m\x1b[K\n");
        out.push_str("\x1b[J");

        print!("{}", out);
        std::io::stdout().flush().ok();
    }
}

impl Drop for Dashboard {
    fn drop(&mut self) {
        if self.is_tty {
            print!("\x1b[?7h\x1b[?25h\x1b[?1049l\n");
            std::io::stdout().flush().ok();
        }
    }
}

#[repr(C)]
struct CudaDeviceInfo {
    name: [u8; 256],
    arch_name: [u8; 64],
    sm_count: i32,
    major: i32,
    minor: i32,
    total_memory: u64,
    recommended_batch_size: u32,
    max_threads_per_block: u32,
}

#[repr(C)]
struct CudaMiningResult {
    found: u32,
    nonce: u32,
    hash: [u8; 32],
}

unsafe extern "C" {
    fn cuda_miner_get_device_count(out_count: *mut i32) -> i32;
    fn cuda_miner_get_device_info(device_id: i32, out_info: *mut CudaDeviceInfo) -> i32;
    fn cuda_miner_init(device_id: i32) -> i32;
    fn cuda_miner_search_device(
        device_id: i32,
        midstate: *const u32,
        header80: *const u8,
        target: *const u8,
        start_nonce: u32,
        batch_size: u32,
        result: *mut CudaMiningResult,
    ) -> i32;
    fn cuda_miner_cleanup_device(device_id: i32);
    fn cuda_miner_cleanup();
}

fn c_str_to_string(bytes: &[u8]) -> String {
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..len]).to_string()
}

#[derive(Clone)]
enum DeviceBackend {
    Cuda { cuda_id: i32 },
    OpenCl { ocl_info: opencl_backend::OpenClDeviceInfo },
}

#[derive(Clone)]
struct UnifiedDevice {
    name: String,
    arch_name: String,
    sm_count: i32,
    recommended_batch_size: u32,
    backend: DeviceBackend,
}

static RUNNING: AtomicBool = AtomicBool::new(true);

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[inline(always)]
fn ror32(x: u32, n: u32) -> u32 {
    (x >> n) | (x << (32 - n))
}

fn compute_midstate(header64: &[u8]) -> [u32; 8] {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            header64[i * 4],
            header64[i * 4 + 1],
            header64[i * 4 + 2],
            header64[i * 4 + 3],
        ]);
    }

    for i in 16..64 {
        let s0 = ror32(w[i - 15], 7) ^ ror32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        let s1 = ror32(w[i - 2], 17) ^ ror32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let mut a = 0x6a09e667u32;
    let mut b = 0xbb67ae85u32;
    let mut c = 0x3c6ef372u32;
    let mut d = 0xa54ff53au32;
    let mut e = 0x510e527fu32;
    let mut f = 0x9b05688cu32;
    let mut g = 0x1f83d9abu32;
    let mut h = 0x5be0cd19u32;

    for i in 0..64 {
        let s1_val = ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25);
        let ch = (e & f) ^ ((!e) & g);
        let tmp1 = h
            .wrapping_add(s1_val)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0_val = ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let tmp2 = s0_val.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(tmp1);
        d = c;
        c = b;
        b = a;
        a = tmp1.wrapping_add(tmp2);
    }

    [
        0x6a09e667u32.wrapping_add(a),
        0xbb67ae85u32.wrapping_add(b),
        0x3c6ef372u32.wrapping_add(c),
        0xa54ff53au32.wrapping_add(d),
        0x510e527fu32.wrapping_add(e),
        0x9b05688cu32.wrapping_add(f),
        0x1f83d9abu32.wrapping_add(g),
        0x5be0cd19u32.wrapping_add(h),
    ]
}

pub fn diff_to_target_le(diff: f64) -> [u8; 32] {
    let d = if diff <= 0.0000001 { 0.0000001 } else { diff };
    let mut limbs = [0u64; 4];
    limbs[3] = 0x00000000ffff0000u64;

    let mut result_limbs = [0u64; 4];
    let mut remainder = 0.0f64;

    for i in (0..4).rev() {
        let cur = remainder * (18446744073709551616.0f64) + (limbs[i] as f64);
        let q = cur / d;
        if q >= 18446744073709551616.0f64 {
            result_limbs[i] = u64::MAX;
            remainder = 0.0;
        } else {
            let q_u64 = q.floor() as u64;
            result_limbs[i] = q_u64;
            remainder = cur - (q_u64 as f64) * d;
        }
    }

    let mut out = [0u8; 32];
    for (i, limb) in result_limbs.iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    out
}

pub fn bits_to_target_le(bits: u32) -> [u8; 32] {
    let mut target = [0u8; 32];
    let exponent = (bits >> 24) as usize;
    let mantissa = bits & 0x007f_ffff;
    if exponent < 3 {
        return [0xff; 32];
    }
    let start = exponent - 3;
    if start > 29 {
        return [0xff; 32];
    }
    target[start] = (mantissa & 0xff) as u8;
    if start + 1 < 32 {
        target[start + 1] = ((mantissa >> 8) & 0xff) as u8;
    }
    if start + 2 < 32 {
        target[start + 2] = ((mantissa >> 16) & 0xff) as u8;
    }
    target
}

pub fn compute_tru_hash(h80: &[u8; 80]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let h1 = Sha256::digest(h80);
    let mut h2: [u8; 32] = Sha256::digest(&h1).into();
    let mut last32 = ((h2[28] as u32) << 24)
        | ((h2[29] as u32) << 16)
        | ((h2[30] as u32) << 8)
        | (h2[31] as u32);
    last32 = last32.wrapping_add(0x21E8);
    h2[28] = ((last32 >> 24) & 0xff) as u8;
    h2[29] = ((last32 >> 16) & 0xff) as u8;
    h2[30] = ((last32 >> 8) & 0xff) as u8;
    h2[31] = (last32 & 0xff) as u8;
    h2
}

pub fn compare256_le(hash_val: &[u8; 32], target_val: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if hash_val[i] < target_val[i] {
            return true;
        }
        if hash_val[i] > target_val[i] {
            return false;
        }
    }
    true
}

#[derive(Clone, Debug)]
pub struct ActiveJob {
    pub job_id: String,
    pub height: u32,
    pub bits_hex: String,
    pub header80: [u8; 80],
    pub midstate: [u32; 8],
    pub target_le: [u8; 32],
    pub diff: f64,
}

pub fn is_valid_tru_address(addr: &str) -> bool {
    let s = addr.trim();
    if s.len() != 34 || !s.starts_with('T') {
        return false;
    }
    s.chars().all(|c| {
        (c >= '1' && c <= '9')
            || (c >= 'A' && c <= 'H')
            || (c >= 'J' && c <= 'N')
            || (c >= 'P' && c <= 'Z')
            || (c >= 'a' && c <= 'k')
            || (c >= 'm' && c <= 'z')
    })
}

pub struct ShareSubmission {
    pub job_id: String,
    pub nonce: u32,
    pub ntime: u32,
    pub hashrate: f64,
    pub is_block: bool,
    pub height: u32,
}

pub enum PoolFeedback {
    ShareAccepted { diff: f64 },
    ShareRejected { reason: String },
    BlockFound { height: u32 },
    NewDiff { diff: f64 },
    StatusEvent { message: String },
}

fn rand_u32() -> u32 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    (now & 0xffffffff) as u32
}

async fn connect_ws_fast(
    url_str: &str,
) -> Result<
    (
        tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
        tokio_tungstenite::tungstenite::handshake::client::Response,
    ),
    Box<dyn std::error::Error + Send + Sync>,
> {
    let uri: tokio_tungstenite::tungstenite::http::Uri = url_str.parse()?;
    let host = uri.host().unwrap_or("tru.powgrid.xyz");
    let is_tls = uri.scheme_str() == Some("wss");
    let port = uri.port_u16().unwrap_or(if is_tls { 443 } else { 80 });

    let addrs = tokio::net::lookup_host((host, port)).await?;
    let mut sorted_addrs: Vec<std::net::SocketAddr> = Vec::new();
    let mut v6_addrs: Vec<std::net::SocketAddr> = Vec::new();
    for a in addrs {
        if a.is_ipv4() {
            sorted_addrs.push(a);
        } else {
            v6_addrs.push(a);
        }
    }
    sorted_addrs.extend(v6_addrs);

    let mut last_err: Option<String> = None;
    let mut connected_tcp = None;
    for addr in sorted_addrs {
        match tokio::time::timeout(Duration::from_secs(3), tokio::net::TcpStream::connect(addr)).await {
            Ok(Ok(tcp)) => {
                let _ = tcp.set_nodelay(true);
                connected_tcp = Some(tcp);
                break;
            }
            Ok(Err(e)) => last_err = Some(e.to_string()),
            Err(_) => last_err = Some("Connection timed out".to_string()),
        }
    }

    let tcp = connected_tcp.ok_or_else(|| last_err.unwrap_or_else(|| "No address resolved".to_string()))?;
    let res = tokio_tungstenite::client_async_tls(url_str, tcp).await?;
    Ok(res)
}

fn run_wss_client(
    wss_url: String,
    wallet_addr: String,
    worker_name: String,
    job_slot: Arc<RwLock<Option<ActiveJob>>>,
    mut share_rx: tokio::sync::mpsc::UnboundedReceiver<ShareSubmission>,
    feedback_tx: Sender<PoolFeedback>,
) {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            let _ = feedback_tx.send(PoolFeedback::StatusEvent {
                message: format!("[{}] ❌ Failed to create async runtime: {}", get_time_str(), e),
            });
            return;
        }
    };

    rt.block_on(async move {
        use futures_util::{SinkExt, StreamExt};
        let user_pass = format!("{}.{}", wallet_addr, worker_name);

        while RUNNING.load(Ordering::Relaxed) {
            let _ = feedback_tx.send(PoolFeedback::StatusEvent {
                message: format!("[{}] 🌐 Connecting to Stratum WSS: {}", get_time_str(), wss_url),
            });

            match connect_ws_fast(&wss_url).await {
                Ok((ws_stream, _)) => {
                    let _ = feedback_tx.send(PoolFeedback::StatusEvent {
                        message: format!("[{}] ⚡ Connected to WSS Stratum! Subscribing...", get_time_str()),
                    });

                    let (mut write, mut read) = ws_stream.split();

                    // 1. mining.subscribe
                    let sub_req = json!({
                        "id": 1,
                        "method": "mining.subscribe",
                        "params": ["powgrid-tru-gpu-miner/1.2", null]
                    });
                    if write.send(tokio_tungstenite::tungstenite::Message::Text(sub_req.to_string().into())).await.is_err() {
                        tokio::time::sleep(Duration::from_secs(2)).await;
                        continue;
                    }

                    // 2. mining.authorize
                    let auth_req = json!({
                        "id": 2,
                        "method": "mining.authorize",
                        "params": [user_pass.clone(), "x"]
                    });
                    if write.send(tokio_tungstenite::tungstenite::Message::Text(auth_req.to_string().into())).await.is_err() {
                        tokio::time::sleep(Duration::from_secs(2)).await;
                        continue;
                    }

                    let current_diff = Arc::new(RwLock::new(1.0f64));
                    let diff_clone = Arc::clone(&current_diff);
                    let job_slot_clone = Arc::clone(&job_slot);
                    let feedback_clone = feedback_tx.clone();
                    let pending_submits = Arc::new(Mutex::new(HashMap::<u64, (bool, u32)>::new()));
                    let pending_submits_read = Arc::clone(&pending_submits);
                    let (pong_tx, mut pong_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();

                    // Purge any stale shares queued while offline / reconnecting
                    while share_rx.try_recv().is_ok() {}

                    let mut read_task = tokio::spawn(async move {
                        while let Some(msg_res) = read.next().await {
                            match msg_res {
                                Ok(tokio_tungstenite::tungstenite::Message::Text(txt)) => {
                                    if let Ok(v) = serde_json::from_str::<Value>(&txt) {
                                        if let Some(method) = v.get("method").and_then(|m| m.as_str()) {
                                            if method == "mining.set_difficulty" {
                                                if let Some(params) = v.get("params").and_then(|p| p.as_array()) {
                                                    if let Some(d) = params.get(0).and_then(|x| x.as_f64()) {
                                                        if let Ok(mut w) = diff_clone.write() {
                                                            *w = d;
                                                        }
                                                        let _ = feedback_clone.send(PoolFeedback::NewDiff { diff: d });
                                                        if let Ok(mut lock) = job_slot_clone.write() {
                                                            if let Some(ref mut job) = *lock {
                                                                job.diff = d;
                                                                job.target_le = diff_to_target_le(d);
                                                            }
                                                        }
                                                    }
                                                }
                                            } else if method == "mining.notify" {
                                                if let Some(params) = v.get("params").and_then(|p| p.as_array()) {
                                                    let job_id = params.get(0).and_then(|x| x.as_str()).unwrap_or("").to_string();
                                                    let bits_str = params.get(4).and_then(|x| x.as_str()).unwrap_or("1d00ffff").to_string();
                                                    let header_hex = params.get(7).and_then(|x| x.as_str()).unwrap_or("").to_string();

                                                    if let Ok(hbytes) = hex::decode(&header_hex) {
                                                        if hbytes.len() >= 76 {
                                                            let mut h80 = [0u8; 80];
                                                            h80[0..76].copy_from_slice(&hbytes[0..76]);
                                                            let midstate = compute_midstate(&h80[0..64]);
                                                            let d = params
                                                                .get(5)
                                                                .and_then(|x| x.as_f64())
                                                                .or_else(|| diff_clone.read().ok().map(|r| *r))
                                                                .unwrap_or(1.0);
                                                            if let Ok(mut w) = diff_clone.write() {
                                                                *w = d;
                                                            }
                                                            let target_le = diff_to_target_le(d);

                                                            let height = job_id.split('_').next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);

                                                            let job = ActiveJob {
                                                                job_id: job_id.clone(),
                                                                height,
                                                                bits_hex: bits_str,
                                                                header80: h80,
                                                                midstate,
                                                                target_le,
                                                                diff: d,
                                                            };

                                                            if let Ok(mut lock) = job_slot_clone.write() {
                                                                *lock = Some(job);
                                                            }

                                                            let _ = feedback_clone.send(PoolFeedback::StatusEvent {
                                                                message: format!("[{}] ⚡ New block #{} work template (diff: {:.2})", get_time_str(), height, d),
                                                            });
                                                        }
                                                    }
                                                }
                                            } else if method == "client.show_message" {
                                                if let Some(params) = v.get("params").and_then(|p| p.as_array()) {
                                                    if let Some(msg) = params.get(0).and_then(|x| x.as_str()) {
                                                        let upper = msg.to_uppercase();
                                                        if !upper.contains("BLOCK") && !upper.contains("SOLVED") {
                                                            let _ = feedback_clone.send(PoolFeedback::StatusEvent {
                                                                message: format!("[{}] 📢 {}", get_time_str(), msg),
                                                            });
                                                        }
                                                    }
                                                }
                                            }
                                        } else if let Some(id) = v.get("id").and_then(|i| i.as_u64()) {
                                            if id >= 10 {
                                                let result = v.get("result").and_then(|r| r.as_bool()).unwrap_or(false);
                                                let pending = pending_submits_read.lock().ok().and_then(|mut m| m.remove(&id));
                                                if result {
                                                    let server_block = v.get("block_found").and_then(|b| b.as_bool())
                                                        .or_else(|| v.get("blockFound").and_then(|b| b.as_bool()))
                                                        .unwrap_or(false);
                                                    let is_block = server_block || pending.map(|(b, _)| b).unwrap_or(false);
                                                    let height = v.get("height").and_then(|h| h.as_u64()).map(|h| h as u32)
                                                        .or_else(|| pending.map(|(_, h)| h))
                                                        .unwrap_or(0);

                                                    if is_block {
                                                        let _ = feedback_clone.send(PoolFeedback::BlockFound { height });
                                                    } else {
                                                        let d = diff_clone.read().map(|r| *r).unwrap_or(1.0);
                                                        let _ = feedback_clone.send(PoolFeedback::ShareAccepted { diff: d });
                                                    }
                                                } else {
                                                    let reason = v.get("error").map(|e| e.to_string()).unwrap_or_else(|| "rejected".to_string());
                                                    let _ = feedback_clone.send(PoolFeedback::ShareRejected { reason });
                                                }
                                            }
                                        }
                                    }
                                }
                                Ok(tokio_tungstenite::tungstenite::Message::Ping(data)) => {
                                    let _ = pong_tx.send(data.to_vec());
                                }
                                Ok(tokio_tungstenite::tungstenite::Message::Close(_)) => break,
                                Err(_) => break,
                                _ => {}
                            }
                        }
                    });

                    let user_pass_submit = user_pass.clone();
                    let mut req_id = 10u64;
                    tokio::select! {
                        _ = &mut read_task => {
                            let _ = feedback_tx.send(PoolFeedback::StatusEvent {
                                message: format!("[{}] ⚠️ WSS disconnected, reconnecting in 2s...", get_time_str()),
                            });
                            tokio::time::sleep(Duration::from_secs(2)).await;
                        }
                        _ = async {
                            let mut ping_interval = tokio::time::interval(Duration::from_secs(20));
                            ping_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                            ping_interval.tick().await;

                            loop {
                                tokio::select! {
                                    Some(pong_data) = pong_rx.recv() => {
                                        if write.send(tokio_tungstenite::tungstenite::Message::Pong(pong_data.into())).await.is_err() {
                                            break;
                                        }
                                    }
                                    _ = ping_interval.tick() => {
                                        if write.send(tokio_tungstenite::tungstenite::Message::Ping(vec![].into())).await.is_err() {
                                            break;
                                        }
                                    }
                                    share_opt = share_rx.recv() => {
                                        let share = match share_opt {
                                            Some(s) => s,
                                            None => break,
                                        };
                                        req_id += 1;
                                        if let Ok(mut m) = pending_submits.lock() {
                                            if m.len() > 1000 {
                                                m.clear();
                                            }
                                            m.insert(req_id, (share.is_block, share.height));
                                        }
                                        let submit_msg = json!({
                                            "id": req_id,
                                            "method": "mining.submit",
                                            "params": [
                                                user_pass_submit,
                                                share.job_id,
                                                "",
                                                format!("0x{:08x}", share.ntime),
                                                format!("0x{:08x}", share.nonce)
                                            ]
                                        });
                                        if write.send(tokio_tungstenite::tungstenite::Message::Text(submit_msg.to_string().into())).await.is_err() {
                                            break;
                                        }
                                    }
                                }
                            }
                        } => {}
                    }
                }
                Err(e) => {
                    let _ = feedback_tx.send(PoolFeedback::StatusEvent {
                        message: format!("[{}] ❌ WSS connect failed: {}, retrying in 2s...", get_time_str(), e),
                    });
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
            }
        }
    });
}

fn run_http_client(
    pool_url: String,
    wallet_addr: String,
    worker_name: String,
    job_slot: Arc<RwLock<Option<ActiveJob>>>,
    mut share_rx: tokio::sync::mpsc::UnboundedReceiver<ShareSubmission>,
    feedback_tx: Sender<PoolFeedback>,
) {
    let template_url = format!("{}/api/pool/template?address={}&worker={}", pool_url, wallet_addr, worker_name);
    let submit_url = format!("{}/api/pool/submit", pool_url);

    let mut last_fetch = Instant::now() - Duration::from_secs(10);
    let mut current_job_id = String::new();

    while RUNNING.load(Ordering::Relaxed) {
        // 1. Check for shares to submit
        while let Ok(share) = share_rx.try_recv() {
            let submit_body = json!({
                "miner": wallet_addr,
                "worker": worker_name,
                "job_id": share.job_id,
                "nonce": share.nonce,
                "ntime": share.ntime,
                "hashrate": share.hashrate,
            });

            match ureq::post(&submit_url).timeout(Duration::from_millis(3000)).send_json(submit_body) {
                Ok(resp) => {
                    if let Ok(val) = resp.into_json::<Value>() {
                        if val["accepted"].as_bool().unwrap_or(false) || val["validShare"].as_bool().unwrap_or(false) {
                            let server_block = val["blockFound"].as_bool().unwrap_or(false) || val["block_found"].as_bool().unwrap_or(false);
                            if server_block || share.is_block {
                                let height = val["height"].as_u64().map(|h| h as u32).unwrap_or(share.height);
                                let _ = feedback_tx.send(PoolFeedback::BlockFound { height });
                            } else {
                                let diff = val["workerDifficulty"].as_f64().unwrap_or(1.0);
                                let _ = feedback_tx.send(PoolFeedback::ShareAccepted { diff });
                            }
                            if let Some(nd) = val["newDifficulty"].as_f64() {
                                let _ = feedback_tx.send(PoolFeedback::NewDiff { diff: nd });
                            }
                        } else {
                            let reason = val["message"].as_str().or(val["error"].as_str()).unwrap_or("rejected").to_string();
                            let _ = feedback_tx.send(PoolFeedback::ShareRejected { reason });
                        }
                    }
                }
                Err(e) => {
                    let _ = feedback_tx.send(PoolFeedback::StatusEvent {
                        message: format!("[{}] ❌ HTTP share submit error: {}", get_time_str(), e),
                    });
                }
            }
        }

        // 2. Poll template every 2s or when job is empty
        if current_job_id.is_empty() || last_fetch.elapsed() >= Duration::from_millis(2000) {
            last_fetch = Instant::now();
            if let Ok(resp) = ureq::get(&template_url).timeout(Duration::from_millis(3000)).call() {
                if let Ok(tpl) = resp.into_json::<Value>() {
                    let new_job_id = tpl["jobId"].as_str().unwrap_or("").to_string();
                    let height = tpl["height"].as_u64().unwrap_or(0) as u32;
                    let bits_hex = tpl["bitsHex"].as_str().unwrap_or("1d00ffff").to_string();
                    let header_hex = tpl["headerHex"].as_str().unwrap_or("").to_string();
                    let worker_diff = tpl["workerDifficulty"].as_f64().unwrap_or(1.0);

                    if new_job_id != current_job_id && !header_hex.is_empty() {
                        current_job_id = new_job_id.clone();
                        if let Ok(hbytes) = hex::decode(&header_hex) {
                            if hbytes.len() >= 76 {
                                let mut h80 = [0u8; 80];
                                h80[0..76].copy_from_slice(&hbytes[0..76]);
                                let midstate = compute_midstate(&h80[0..64]);

                                let mut target_le = diff_to_target_le(worker_diff);
                                if let Some(th) = tpl["targetHex"].as_str() {
                                    if let Ok(tb) = hex::decode(th) {
                                        if tb.len() == 32 {
                                            target_le.copy_from_slice(&tb);
                                        }
                                    }
                                }

                                let job = ActiveJob {
                                    job_id: new_job_id,
                                    height,
                                    bits_hex,
                                    header80: h80,
                                    midstate,
                                    target_le,
                                    diff: worker_diff,
                                };

                                if let Ok(mut lock) = job_slot.write() {
                                    *lock = Some(job);
                                }

                                let _ = feedback_tx.send(PoolFeedback::NewDiff { diff: worker_diff });
                                let _ = feedback_tx.send(PoolFeedback::StatusEvent {
                                    message: format!("[{}] ⚡ New block #{} work template (diff: {:.2})", get_time_str(), height, worker_diff),
                                });
                            }
                        }
                    }
                }
            }
        }

        thread::sleep(Duration::from_millis(10));
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut raw_pool_url = "wss://tru.powgrid.xyz/stratum".to_string();
    let mut wallet_addr = String::new();
    let mut worker_name = "rig-1".to_string();
    let mut selected_devices_arg: Option<String> = None;
    let mut preset = "auto".to_string();
    let mut manual_batch_size: Option<u32> = None;
    let mut manual_watt_limit: Option<u32> = None;
    let mut force_http = false;
    let mut is_hiveos = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pool" | "-o" if i + 1 < args.len() => {
                raw_pool_url = args[i + 1].clone();
                i += 1;
            }
            "--wallet" | "-u" | "--address" if i + 1 < args.len() => {
                wallet_addr = args[i + 1].clone();
                i += 1;
            }
            "--worker" | "-w" if i + 1 < args.len() => {
                worker_name = args[i + 1].clone();
                i += 1;
            }
            "--device" | "--devices" | "--gpu" | "-d" if i + 1 < args.len() => {
                selected_devices_arg = Some(args[i + 1].clone());
                i += 1;
            }
            "--preset" if i + 1 < args.len() => {
                preset = args[i + 1].clone();
                i += 1;
            }
            "--batch-size" if i + 1 < args.len() => {
                if let Ok(bs) = args[i + 1].parse::<u32>() {
                    manual_batch_size = Some(bs);
                }
                i += 1;
            }
            "--watt" | "--power-limit" if i + 1 < args.len() => {
                if let Ok(w) = args[i + 1].parse::<u32>() {
                    manual_watt_limit = Some(w);
                }
                i += 1;
            }
            "--http" => {
                force_http = true;
            }
            "--hiveos" => {
                is_hiveos = true;
            }
            "--help" | "-h" => {
                println!("Usage: powgrid-tru-gpu-miner [OPTIONS]");
                println!("PowGrid TRU GPU Miner v1.3.2 - Algorithm: TRUHash");
                println!("Options:");
                println!("  --pool, -o <URL>          Target Pool URL (default: wss://tru.powgrid.xyz/stratum)");
                println!("  --wallet, -u <ADDR>       TRU payout wallet address (required)");
                println!("  --worker, -w <NAME>       Worker/Rig name (default: rig-1)");
                println!("  --gpu, --devices <0,1..>  GPU device indexes (default: all detected NVIDIA / AMD GPUs)");
                println!("  --preset <NAME>           Tuning preset: auto | extreme | max | balanced");
                println!("  --batch-size <NUM>        Override nonce batch size (e.g. 33554432)");
                println!("  --watt <NUM>              Cap GPU power consumption in Watts (e.g. 150)");
                println!("  --http                    Force HTTP REST long-polling instead of WSS");
                println!("  --help, -h                Print this help message and exit");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    if wallet_addr.is_empty() {
        eprintln!("\x1b[1;31m[ERROR]\x1b[0m TRU wallet address is required!");
        eprintln!("Specify with: --wallet <ADDR> or -u <ADDR> (e.g. -u <YOUR_TRU_WALLET_ADDRESS>)");
        std::process::exit(1);
    }

    if let Some(pos) = wallet_addr.find('.') {
        worker_name = wallet_addr[pos + 1..].to_string();
        wallet_addr = wallet_addr[..pos].to_string();
    }

    if worker_name == "%WORKER_NAME%" || worker_name.is_empty() {
        worker_name = env::var("WORKER_NAME").unwrap_or_else(|_| "rig-1".to_string());
    }

    let raw_addr_to_check = if wallet_addr.starts_with("solo:") {
        &wallet_addr[5..]
    } else {
        &wallet_addr
    };

    if !is_valid_tru_address(raw_addr_to_check) {
        eprintln!("\x1b[1;31m[ERROR]\x1b[0m '{}' is NOT a valid TRU Base58 address!", wallet_addr);
        eprintln!("TRU addresses must start with 'T', be 34 characters long, and contain only Base58 characters (no '0', 'O', 'I', or 'l').");
        std::process::exit(1);
    }

    let mut detected_cuda_count: i32 = 0;
    unsafe {
        let _ = cuda_miner_get_device_count(&mut detected_cuda_count);
    }
    if detected_cuda_count < 0 {
        detected_cuda_count = 0;
    }

    let all_ocl_devices = opencl_backend::get_opencl_devices();
    let filtered_ocl_devices: Vec<opencl_backend::OpenClDeviceInfo> = all_ocl_devices
        .into_iter()
        .filter(|d| detected_cuda_count == 0 || !d.is_nvidia)
        .collect();

    let mut all_available_devices: Vec<UnifiedDevice> = Vec::new();

    for cuda_id in 0..detected_cuda_count {
        let mut dev_info = CudaDeviceInfo {
            name: [0; 256],
            arch_name: [0; 64],
            sm_count: 0,
            major: 0,
            minor: 0,
            total_memory: 0,
            recommended_batch_size: 33_554_432,
            max_threads_per_block: 256,
        };
        let dev_info_ok = unsafe { cuda_miner_get_device_info(cuda_id, &mut dev_info) == 0 };
        let dev_name = if dev_info_ok {
            c_str_to_string(&dev_info.name)
        } else {
            format!("CUDA GPU #{}", cuda_id)
        };
        let arch_name = if dev_info_ok {
            c_str_to_string(&dev_info.arch_name)
        } else {
            "CUDA".to_string()
        };
        let recommended_batch_size = if dev_info_ok {
            dev_info.recommended_batch_size
        } else {
            33_554_432
        };
        all_available_devices.push(UnifiedDevice {
            name: dev_name,
            arch_name,
            sm_count: dev_info.sm_count,
            recommended_batch_size,
            backend: DeviceBackend::Cuda { cuda_id },
        });
    }

    for ocl_info in filtered_ocl_devices {
        let arch_name = if ocl_info.vendor.to_lowercase().contains("amd")
            || ocl_info.vendor.to_lowercase().contains("advanced micro")
        {
            "OpenCL AMD".to_string()
        } else if ocl_info.vendor.to_lowercase().contains("intel") {
            "OpenCL Intel".to_string()
        } else {
            format!("OpenCL {}", ocl_info.vendor)
        };

        let recommended_batch_size = if ocl_info.compute_units >= 60 {
            33_554_432
        } else if ocl_info.compute_units >= 30 {
            16_777_216
        } else {
            8_388_608
        };

        all_available_devices.push(UnifiedDevice {
            name: ocl_info.device_name.clone(),
            arch_name,
            sm_count: ocl_info.compute_units as i32,
            recommended_batch_size,
            backend: DeviceBackend::OpenCl { ocl_info },
        });
    }

    if all_available_devices.is_empty() {
        all_available_devices.push(UnifiedDevice {
            name: "CUDA GPU #0".to_string(),
            arch_name: "CUDA".to_string(),
            sm_count: 0,
            recommended_batch_size: 33_554_432,
            backend: DeviceBackend::Cuda { cuda_id: 0 },
        });
    }

    let active_indexes: Vec<usize> = match selected_devices_arg {
        Some(ref s) => {
            let s_lower = s.to_lowercase();
            if s_lower == "all" || s_lower == "*" {
                (0..all_available_devices.len()).collect()
            } else {
                let mut list = Vec::new();
                for part in s.split(',') {
                    let trimmed = part.trim();
                    if let Ok(id) = trimmed.parse::<usize>() {
                        if id < all_available_devices.len() {
                            if !list.contains(&id) {
                                list.push(id);
                            }
                        }
                    }
                }
                if list.is_empty() {
                    vec![0]
                } else {
                    list
                }
            }
        }
        None => (0..all_available_devices.len()).collect(),
    };

    let mut gpu_telemetry_list = Vec::new();
    let mut gpu_shared_states = Vec::new();

    for &idx in &active_indexes {
        let dev = &all_available_devices[idx];
        let batch_size: u32 = match manual_batch_size {
            Some(bs) => bs,
            None => match preset.to_lowercase().as_str() {
                "extreme" => 134_217_728,
                "max" => 67_108_864,
                "balanced" => 33_554_432,
                _ => dev.recommended_batch_size,
            },
        };

        let nvml = match dev.backend {
            DeviceBackend::Cuda { cuda_id } => {
                let mon = NvmlMonitor::new(cuda_id as u32);
                if let Some(w) = manual_watt_limit {
                    mon.apply_power_limit(cuda_id as u32, w);
                }
                mon
            }
            DeviceBackend::OpenCl { .. } => NvmlMonitor {
                device: std::ptr::null_mut(),
                fn_temp: None,
                fn_power: None,
                fn_set_power: None,
            },
        };

        let hashes_counter = Arc::new(AtomicU64::new(0));

        gpu_telemetry_list.push(GpuDeviceTelemetry {
            device_id: idx as i32,
            name: dev.name.clone(),
            arch_name: dev.arch_name.clone(),
            sm_count: dev.sm_count,
            batch_size,
            nvml,
            current_mhs: 0.0,
            total_hashes: 0,
            last_tick_hashes: 0,
        });

        gpu_shared_states.push((idx, dev.backend.clone(), batch_size, hashes_counter));
    }

    let mut clean_pool_url = raw_pool_url.trim().to_string();
    for prefix in &["stratum+ssl://", "stratum+tcp://", "stratum+ws://", "stratum+wss://", "stratum://"] {
        if clean_pool_url.starts_with(prefix) {
            clean_pool_url = clean_pool_url[prefix.len()..].to_string();
            break;
        }
    }

    let is_wss = !force_http && (clean_pool_url.starts_with("wss://") || clean_pool_url.starts_with("ws://") || clean_pool_url.contains("powgrid.xyz"));
    let active_pool_url = if is_wss {
        if clean_pool_url.starts_with("http://") {
            clean_pool_url.replace("http://", "ws://").trim_end_matches('/').to_string() + "/stratum"
        } else if clean_pool_url.starts_with("https://") {
            clean_pool_url.replace("https://", "wss://").trim_end_matches('/').to_string() + "/stratum"
        } else if clean_pool_url.starts_with("ws://") || clean_pool_url.starts_with("wss://") {
            if !clean_pool_url.ends_with("/stratum") {
                format!("{}/stratum", clean_pool_url.trim_end_matches('/'))
            } else {
                clean_pool_url
            }
        } else {
            let host_part = clean_pool_url.trim_end_matches('/');
            if host_part.ends_with("/stratum") {
                format!("wss://{}", host_part)
            } else {
                format!("wss://{}/stratum", host_part)
            }
        }
    } else {
        clean_pool_url
    };

    ctrlc::set_handler(move || {
        RUNNING.store(false, Ordering::SeqCst);
        print!("\x1b[?7h\x1b[?25h\x1b[?1049l\n");
        std::io::stdout().flush().ok();
    })
    .ok();

    let mut dashboard = Dashboard::new(
        gpu_telemetry_list,
        preset,
        active_pool_url.clone(),
        wallet_addr.clone(),
        worker_name.clone(),
        true,
        is_hiveos,
    );

    if let Some(w) = manual_watt_limit {
        dashboard.set_power_cap(w);
    }

    dashboard.add_event(format!(
        "[{}] 🚀 Initialized PowGrid TRU GPU Miner v1.3.2 ({} GPU{} active)",
        get_time_str(),
        dashboard.gpus.len(),
        if dashboard.gpus.len() > 1 { "s" } else { "" }
    ));

    let job_slot = Arc::new(RwLock::new(None::<ActiveJob>));
    let (share_tx, share_rx) = tokio::sync::mpsc::unbounded_channel::<ShareSubmission>();
    let (feedback_tx, feedback_rx) = channel::<PoolFeedback>();

    let net_job_slot = Arc::clone(&job_slot);
    let net_wallet = wallet_addr.clone();
    let net_worker = worker_name.clone();
    let net_url = active_pool_url.clone();

    if is_wss {
        thread::spawn(move || {
            run_wss_client(net_url, net_wallet, net_worker, net_job_slot, share_rx, feedback_tx);
        });
    } else {
        thread::spawn(move || {
            run_http_client(net_url, net_wallet, net_worker, net_job_slot, share_rx, feedback_tx);
        });
    }

    let worker_nonce_seed: u32 = {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&worker_name, &mut hasher);
        std::hash::Hasher::finish(&hasher) as u32
    };

    let nonce_counter = Arc::new(AtomicU32::new(rand_u32() ^ worker_nonce_seed));
    let mut worker_handles = Vec::new();

    for (dev_idx, backend, batch_size, hashes_counter) in gpu_shared_states.iter() {
        let dev_idx = *dev_idx;
        let backend = backend.clone();
        let batch_size = *batch_size;
        let thread_job_slot = Arc::clone(&job_slot);
        let thread_nonce_counter = Arc::clone(&nonce_counter);
        let thread_hashes_counter = Arc::clone(hashes_counter);
        let thread_share_tx = share_tx.clone();

        match backend {
            DeviceBackend::Cuda { cuda_id } => {
                let handle = thread::Builder::new()
                    .name(format!("cuda-{}", cuda_id))
                    .spawn(move || {
                        let init_code = unsafe { cuda_miner_init(cuda_id) };
                        if init_code != 0 {
                            eprintln!("Failed to initialize CUDA miner on device {} (code {})", cuda_id, init_code);
                            return;
                        }

                        let mut local_job_id = String::new();
                        let mut local_height = 0u32;
                        let mut local_bits_hex = "1d00ffff".to_string();
                        let mut local_h80 = [0u8; 80];
                        let mut local_job_base_ntime = 0u32;
                        let mut local_midstate = [0u32; 8];
                        let mut local_target = [0xffu8; 32];
                        let mut local_diff = 0.0f64;
                        let mut local_job_received_at = Instant::now();
                        let mut local_submitted_nonces = HashSet::<u32>::new();

                        while RUNNING.load(Ordering::Relaxed) {
                            {
                                if let Ok(lock) = thread_job_slot.read() {
                                    if let Some(ref j) = *lock {
                                        let job_changed = j.job_id != local_job_id;
                                        let diff_changed = (j.diff - local_diff).abs() > 0.0001;

                                        if job_changed || diff_changed {
                                            if job_changed {
                                                local_job_id = j.job_id.clone();
                                                local_height = j.height;
                                                local_bits_hex = j.bits_hex.clone();
                                                local_h80 = j.header80;
                                                local_midstate = j.midstate;
                                                local_job_base_ntime = u32::from_le_bytes([j.header80[68], j.header80[69], j.header80[70], j.header80[71]]);
                                                local_job_received_at = Instant::now();
                                                local_submitted_nonces.clear();
                                            }
                                            local_target = j.target_le;
                                            local_diff = j.diff;
                                        }
                                    }
                                }
                            }

                            if local_job_id.is_empty() {
                                thread::sleep(Duration::from_millis(30));
                                continue;
                            }

                            let elapsed_sec = local_job_received_at.elapsed().as_secs() as u32;
                            let effective_ntime = local_job_base_ntime.saturating_add(elapsed_sec.min(55));
                            let active_cur_ntime = u32::from_le_bytes([local_h80[68], local_h80[69], local_h80[70], local_h80[71]]);
                            if effective_ntime != active_cur_ntime {
                                local_h80[68..72].copy_from_slice(&effective_ntime.to_le_bytes());
                                local_submitted_nonces.clear();
                            }

                            let start_nonce = thread_nonce_counter.fetch_add(batch_size, Ordering::Relaxed);

                            let mut res = CudaMiningResult {
                                found: 0,
                                nonce: 0,
                                hash: [0; 32],
                            };

                            unsafe {
                                cuda_miner_search_device(
                                    cuda_id,
                                    local_midstate.as_ptr(),
                                    local_h80.as_ptr(),
                                    local_target.as_ptr(),
                                    start_nonce,
                                    batch_size,
                                    &mut res,
                                );
                            }

                            thread_hashes_counter.fetch_add(batch_size as u64, Ordering::Relaxed);

                            if res.found != 0 && local_submitted_nonces.insert(res.nonce) {
                                let mut check_h80 = local_h80;
                                check_h80[76..80].copy_from_slice(&res.nonce.to_le_bytes());
                                let verified_hash = compute_tru_hash(&check_h80);

                                if compare256_le(&verified_hash, &local_target) {
                                    let current_ntime = u32::from_le_bytes([local_h80[68], local_h80[69], local_h80[70], local_h80[71]]);
                                    let bits = u32::from_str_radix(&local_bits_hex, 16)
                                        .unwrap_or_else(|_| u32::from_le_bytes([local_h80[72], local_h80[73], local_h80[74], local_h80[75]]));
                                    let block_target = bits_to_target_le(bits);
                                    let is_block = compare256_le(&verified_hash, &block_target);

                                    let _ = thread_share_tx.send(ShareSubmission {
                                        job_id: local_job_id.clone(),
                                        nonce: res.nonce,
                                        ntime: current_ntime,
                                        hashrate: 0.0,
                                        is_block,
                                        height: local_height,
                                    });
                                }
                            }
                        }

                        unsafe { cuda_miner_cleanup_device(cuda_id) };
                    });

                if let Ok(h) = handle {
                    worker_handles.push(h);
                }
            }
            DeviceBackend::OpenCl { ocl_info } => {
                let handle = thread::Builder::new()
                    .name(format!("opencl-{}", dev_idx))
                    .spawn(move || {
                        let mut ocl_worker = match opencl_backend::OpenClWorker::new(&ocl_info) {
                            Ok(w) => w,
                            Err(e) => {
                                eprintln!("Failed to initialize OpenCL worker on device '{}': {}", ocl_info.device_name, e);
                                return;
                            }
                        };

                        let mut local_job_id = String::new();
                        let mut local_height = 0u32;
                        let mut local_bits_hex = "1d00ffff".to_string();
                        let mut local_h80 = [0u8; 80];
                        let mut local_job_base_ntime = 0u32;
                        let mut local_midstate = [0u32; 8];
                        let mut local_target = [0xffu8; 32];
                        let mut local_diff = 0.0f64;
                        let mut local_job_received_at = Instant::now();
                        let mut local_submitted_nonces = HashSet::<u32>::new();

                        while RUNNING.load(Ordering::Relaxed) {
                            {
                                if let Ok(lock) = thread_job_slot.read() {
                                    if let Some(ref j) = *lock {
                                        let job_changed = j.job_id != local_job_id;
                                        let diff_changed = (j.diff - local_diff).abs() > 0.0001;

                                        if job_changed || diff_changed {
                                            if job_changed {
                                                local_job_id = j.job_id.clone();
                                                local_height = j.height;
                                                local_bits_hex = j.bits_hex.clone();
                                                local_h80 = j.header80;
                                                local_midstate = j.midstate;
                                                local_job_base_ntime = u32::from_le_bytes([j.header80[68], j.header80[69], j.header80[70], j.header80[71]]);
                                                local_job_received_at = Instant::now();
                                                local_submitted_nonces.clear();
                                            }
                                            local_target = j.target_le;
                                            local_diff = j.diff;
                                        }
                                    }
                                }
                            }

                            if local_job_id.is_empty() {
                                thread::sleep(Duration::from_millis(30));
                                continue;
                            }

                            let elapsed_sec = local_job_received_at.elapsed().as_secs() as u32;
                            let effective_ntime = local_job_base_ntime.saturating_add(elapsed_sec.min(55));
                            let active_cur_ntime = u32::from_le_bytes([local_h80[68], local_h80[69], local_h80[70], local_h80[71]]);
                            if effective_ntime != active_cur_ntime {
                                local_h80[68..72].copy_from_slice(&effective_ntime.to_le_bytes());
                                local_submitted_nonces.clear();
                            }

                            let start_nonce = thread_nonce_counter.fetch_add(batch_size, Ordering::Relaxed);

                            let search_res = ocl_worker.search(
                                &local_midstate,
                                &local_h80,
                                &local_target,
                                start_nonce,
                                batch_size,
                            );

                            thread_hashes_counter.fetch_add(batch_size as u64, Ordering::Relaxed);

                            if let Ok(Some((nonce, _hash))) = search_res {
                                if local_submitted_nonces.insert(nonce) {
                                    let mut check_h80 = local_h80;
                                    check_h80[76..80].copy_from_slice(&nonce.to_le_bytes());
                                    let verified_hash = compute_tru_hash(&check_h80);

                                    if compare256_le(&verified_hash, &local_target) {
                                        let current_ntime = u32::from_le_bytes([local_h80[68], local_h80[69], local_h80[70], local_h80[71]]);
                                        let bits = u32::from_str_radix(&local_bits_hex, 16)
                                            .unwrap_or_else(|_| u32::from_le_bytes([local_h80[72], local_h80[73], local_h80[74], local_h80[75]]));
                                        let block_target = bits_to_target_le(bits);
                                        let is_block = compare256_le(&verified_hash, &block_target);

                                        let _ = thread_share_tx.send(ShareSubmission {
                                            job_id: local_job_id.clone(),
                                            nonce,
                                            ntime: current_ntime,
                                            hashrate: 0.0,
                                            is_block,
                                            height: local_height,
                                        });
                                    }
                                }
                            }
                        }
                    });

                if let Ok(h) = handle {
                    worker_handles.push(h);
                }
            }
        }
    }

    let mut blocks_found = 0u32;
    let mut last_stat_tick = Instant::now();
    let mut last_job_id_seen = String::new();
    let mut current_height = 0u32;
    let mut current_bits_hex = "1d00ffff".to_string();

    while RUNNING.load(Ordering::Relaxed) {
        while let Ok(fb) = feedback_rx.try_recv() {
            match fb {
                PoolFeedback::ShareAccepted { diff } => {
                    dashboard.accepted_shares += 1;
                    dashboard.add_event(format!(
                        "[{}] 🟢 Share ACCEPTED (diff {:.2})",
                        get_time_str(),
                        diff
                    ));
                }
                PoolFeedback::ShareRejected { reason } => {
                    dashboard.rejected_shares += 1;
                    dashboard.add_event(format!(
                        "[{}] 🔴 Share REJECTED: {}",
                        get_time_str(),
                        reason
                    ));
                }
                PoolFeedback::BlockFound { height } => {
                    blocks_found += 1;
                    dashboard.accepted_shares += 1;
                    dashboard.add_event(format!(
                        "[{}] 🏆 SOLVED BLOCK #{}",
                        get_time_str(),
                        height
                    ));
                }
                PoolFeedback::NewDiff { diff } => {
                    dashboard.worker_diff = diff;
                }
                PoolFeedback::StatusEvent { message } => {
                    dashboard.add_event(message);
                }
            }
        }

        if let Ok(lock) = job_slot.read() {
            if let Some(ref j) = *lock {
                if j.job_id != last_job_id_seen {
                    last_job_id_seen = j.job_id.clone();
                    current_height = j.height;
                    current_bits_hex = j.bits_hex.clone();
                    nonce_counter.store(rand_u32() ^ worker_nonce_seed, Ordering::Relaxed);
                }
                dashboard.worker_diff = j.diff;
            }
        }

        let dt = last_stat_tick.elapsed().as_secs_f64();
        if dt >= 0.5 {
            for (idx, (_, _, _, hashes_counter)) in gpu_shared_states.iter().enumerate() {
                if let Some(gpu) = dashboard.gpus.get_mut(idx) {
                    let cur_h = hashes_counter.load(Ordering::Relaxed);
                    let diff_h = cur_h.saturating_sub(gpu.last_tick_hashes);
                    gpu.current_mhs = (diff_h as f64 / dt) / 1_000_000.0;
                    gpu.last_tick_hashes = cur_h;
                    gpu.total_hashes = cur_h;
                }
            }
            last_stat_tick = Instant::now();

            let total_mhs: f64 = dashboard.gpus.iter().map(|g| g.current_mhs).sum();
            dashboard.render(current_height, &current_bits_hex, blocks_found, false);

            let hs_list: Vec<u64> = dashboard.gpus.iter().map(|g| (g.current_mhs * 1_000_000.0) as u64).collect();
            let hs_json = serde_json::to_string(&hs_list).unwrap_or_else(|_| "[]".to_string());

            let _ = std::fs::write(
                "/tmp/powgrid_tru_miner_stats.json",
                format!(
                    r#"{{"uptime":{},"hashrate_avg":{},"hs":{},"accepted":{},"rejected":{}}}"#,
                    dashboard.start_time.elapsed().as_secs(),
                    (total_mhs * 1_000_000.0) as u64,
                    hs_json,
                    dashboard.accepted_shares,
                    dashboard.rejected_shares
                ),
            );
        }

        thread::sleep(Duration::from_millis(50));
    }

    for h in worker_handles {
        let _ = h.join();
    }
    unsafe { cuda_miner_cleanup() };

    if dashboard.is_tty {
        print!("\x1b[?7h\x1b[?25h\x1b[?1049l\n");
        std::io::stdout().flush().ok();
    }
    println!("PowGrid TRU GPU miner stopped cleanly.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_col_row_exact_fit() {
        let left = format!("\x1b[90mPrimary GPU:\x1b[0m \x1b[1;37m{}\x1b[0m", "RTX 5060 Laptop (26 SMs)");
        let right = format!("\x1b[90mAlgorithm  :\x1b[0m \x1b[1;32m{}\x1b[0m", "TRUHash (Blackwell)");
        let row = two_col_row(&left, 39, &right, 74);
        let trimmed = row.trim_end_matches("\x1b[K\n");
        assert_eq!(visible_len(trimmed), 78);
        assert!(trimmed.starts_with("│ "));
        assert!(trimmed.ends_with(" │"));
    }

    #[test]
    fn test_shares_and_blocks_row() {
        let left = format!("\x1b[90mShares (Acc)   :\x1b[0m \x1b[1;32m{}\x1b[0m ({:.1}%)", 20, 100.0);
        let right = format!("\x1b[90mRejected    :\x1b[0m \x1b[1;32m{}\x1b[0m ({:.1}%)", 0, 0.0);
        let row = two_col_row(&left, 39, &right, 74);
        let trimmed = row.trim_end_matches("\x1b[K\n");
        assert_eq!(visible_len(trimmed), 78);
        assert!(trimmed.starts_with("│ "));
        assert!(trimmed.ends_with(" │"));

        let left_b = format!("\x1b[90mBlocks Found   :\x1b[0m \x1b[1;37m★ 0\x1b[0m \x1b[90m(hunting)\x1b[0m");
        let right_b = format!("\x1b[90mBlock Height:\x1b[0m \x1b[1;36m#{}\x1b[0m", 27732);
        let row_b = two_col_row(&left_b, 39, &right_b, 74);
        let trimmed_b = row_b.trim_end_matches("\x1b[K\n");
        assert_eq!(visible_len(trimmed_b), 78);
        assert!(trimmed_b.starts_with("│ "));
        assert!(trimmed_b.ends_with(" │"));
    }

    #[test]
    fn test_two_col_row_defensive_clamp() {
        // Very long left and right strings
        let left = "Primary GPU: An Extremely Long GPU Name That Definitely Exceeds 39 Characters";
        let right = "Algorithm  : Super Long Algorithm Name Exceeding Whatever Space Is Left";
        let row = two_col_row(left, 39, right, 74);
        let trimmed = row.trim_end_matches("\x1b[K\n");
        assert_eq!(visible_len(trimmed), 78);
        assert!(trimmed.starts_with("│ "));
        assert!(trimmed.ends_with(" │"));
    }

    #[test]
    fn test_full_box_symmetry() {
        let border_top = format!("╭{}╮", "─".repeat(76));
        let border_div = format!("├{}┤", "─".repeat(76));
        let border_bot = format!("╰{}╯", "─".repeat(76));

        assert_eq!(visible_len(&border_top), 78);
        assert_eq!(visible_len(&border_div), 78);
        assert_eq!(visible_len(&border_bot), 78);

        let lines = vec![
            box_row("\x1b[1;36m► POWGRID TRU ($TRU) HIGH-PERFORMANCE TRUHASH GPU MINER v1.3.2\x1b[0m", 74),
            two_col_row("\x1b[90mPool:\x1b[0m \x1b[1;37mwss://tru.powgrid.xyz/stratum\x1b[0m", 39, "\x1b[90mStatus:\x1b[0m \x1b[1;32m● ONLINE (Connected)\x1b[0m", 74),
            two_col_row("\x1b[90mWallet:\x1b[0m \x1b[36mTRU1111111...1111\x1b[0m", 39, "\x1b[90mWorker:\x1b[0m \x1b[1;33mlaptop\x1b[0m \x1b[90m(PPLNS Pool)\x1b[0m", 74),
            box_row("\x1b[1;35mHARDWARE & GPU ENGINE CONFIGURATION (1 GPUs Active)\x1b[0m", 74),
            two_col_row("\x1b[90mPrimary GPU:\x1b[0m \x1b[1;37mRTX 5060 Laptop (26 SMs)\x1b[0m", 39, "\x1b[90mAlgorithm  :\x1b[0m \x1b[1;32mTRUHash (Blackwell)\x1b[0m", 74),
            two_col_row("\x1b[90mTelemetry  :\x1b[0m \x1b[1;32m68\x1b[0m°C | 45W | \x1b[1;36m34.2\x1b[0m MH/W", 39, "\x1b[90mBatch Size :\x1b[0m \x1b[1;33m33M\x1b[0m (auto)", 74),
            box_row("\x1b[1;33mMINING TELEMETRY\x1b[0m", 74),
            two_col_row("\x1b[90mHashrate(Now)  :\x1b[0m \x1b[1;32m1540.49 MH/s\x1b[0m", 39, "\x1b[90mUptime      :\x1b[0m \x1b[1;37m00:00:03\x1b[0m", 74),
            two_col_row("\x1b[90mHashrate(Avg)  :\x1b[0m \x1b[1;32m1169.95 MH/s\x1b[0m", 39, "\x1b[90mTarget Diff :\x1b[0m \x1b[1;36m1.80 (Vardiff)\x1b[0m", 74),
            two_col_row("\x1b[90mShares (Acc)   :\x1b[0m \x1b[1;32m20\x1b[0m (100.0%)", 39, "\x1b[90mRejected    :\x1b[0m \x1b[1;32m0\x1b[0m (0.0%)", 74),
            two_col_row("\x1b[90mBlocks Found   :\x1b[0m \x1b[1;37m★ 0\x1b[0m \x1b[90m(hunting)\x1b[0m", 39, "\x1b[90mBlock Height:\x1b[0m \x1b[1;36m#27732\x1b[0m", 74),
            box_row("\x1b[1;34mRECENT MINING & NETWORK EVENTS\x1b[0m", 74),
            box_row("[18:54:54] 🟢 Share ACCEPTED (diff 1.80)", 74),
            box_row("[18:54:54] ⚡ New block #27732 work template (diff: 1.80)", 74),
            box_row("[18:54:53] ⚡ Connected to WSS Stratum! Subscribing...", 74),
            box_row("[18:54:53] 🌐 Connecting to Stratum WSS: wss://tru.powgrid.xyz/stratum", 74),
            box_row("[18:54:53] 🚀 Initialized PowGrid TRU GPU Miner v1.3.2 (1 GPU active)", 74),
        ];

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim_end_matches("\x1b[K\n");
            let vlen = visible_len(trimmed);
            assert_eq!(vlen, 78, "Line {} has wrong visible length {}: {:?}", i, vlen, trimmed);
            assert!(trimmed.starts_with("│ "), "Line {} does not start with '│ ': {:?}", i, trimmed);
            assert!(trimmed.ends_with(" │"), "Line {} does not end with ' │': {:?}", i, trimmed);
        }
    }
}
