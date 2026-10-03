use std::time::Instant;


#[repr(C)]
struct CudaMiningResult {
    found: u32,
    nonce: u32,
    hash: [u8; 32],
}

unsafe extern "C" {
    fn cuda_miner_init(device_id: i32) -> i32;
    fn cuda_miner_search(
        midstate: *const u32,
        header80: *const u8,
        target: *const u8,
        start_nonce: u32,
        batch_size: u32,
        result: *mut CudaMiningResult,
    ) -> i32;
    fn cuda_miner_cleanup();
}

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

fn rpc_call(url: &str, cookie: &str, method: &str, params: serde_json::Value) -> Result<serde_json::Value, String> {
    let payload = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    });
    let resp = ureq::post(url)
        .set("Authorization", &format!("Bearer {}", cookie))
        .set("Content-Type", "application/json")
        .timeout(std::time::Duration::from_secs(5))
        .send_json(payload)
        .map_err(|e| format!("RPC error: {}", e))?;

    let val: serde_json::Value = resp.into_json().map_err(|e| format!("JSON decode error: {}", e))?;
    Ok(val)
}

fn main() {
    let rpc_url = std::env::var("NODE_RPC").unwrap_or_else(|_| "http://127.0.0.1:21842/rpc".to_string());
    let cookie_path = std::env::var("COOKIE_PATH").unwrap_or_else(|_| "/tmp/tru_sandbox_ncft/node_data/.rpc-cookie-21832".to_string());
    let miner_wallet = std::env::var("WALLET").unwrap_or_else(|_| "TRU88888rDZc2EAWLBnuc5cY3CUNRd7zWk".to_string());
    let target_height: u64 = std::env::var("TARGET_HEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(101);

    println!("===============================================================");
    println!("⚡ TRU High-Speed Solo GPU Miner (Direct Node RPC, Zero Pool)");
    println!("🎯 Target Height : #{}", target_height);
    println!("👛 Payout Wallet : {}", miner_wallet);
    println!("📡 Node RPC      : {}", rpc_url);
    println!("===============================================================");

    let cookie = match std::fs::read_to_string(&cookie_path) {
        Ok(c) => c.trim().to_string(),
        Err(e) => {
            eprintln!("❌ Failed to read cookie at {}: {}", cookie_path, e);
            std::process::exit(1);
        }
    };

    let init_res = unsafe { cuda_miner_init(0) };
    if init_res != 0 {
        eprintln!("❌ Failed to initialize CUDA miner on device 0 (code {})", init_res);
        std::process::exit(1);
    }
    println!("🚀 [CUDA Init] RTX 5060 GPU engine armed and ready!");

    let batch_size: u32 = 33_554_432;
    let mut total_blocks_found = 0u64;
    let global_start = Instant::now();

    loop {
        // 1. Fetch current block height
        let count_val = match rpc_call(&rpc_url, &cookie, "getblockcount", serde_json::json!([])) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("⚠️ getblockcount error: {}. Retrying in 1s...", e);
                std::thread::sleep(std::time::Duration::from_secs(1));
                continue;
            }
        };
        let current_h = count_val.get("result").and_then(|v| v.as_u64()).unwrap_or(0);
        if current_h >= target_height {
            println!("\n🎉 [TARGET REACHED!] Blockchain height is #{} >= #{}.", current_h, target_height);
            break;
        }

        // 2. Fetch canonical browser miner work from TRU node
        let tmpl_val = match rpc_call(&rpc_url, &cookie, "getblocktemplate", serde_json::json!({"browserMinerAddress": miner_wallet})) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("⚠️ getblocktemplate error: {}. Retrying in 1s...", e);
                std::thread::sleep(std::time::Duration::from_secs(1));
                continue;
            }
        };

        let result = match tmpl_val.get("result") {
            Some(r) if r.is_object() => r,
            _ => {
                eprintln!("⚠️ Invalid getblocktemplate response: {:?}", tmpl_val);
                std::thread::sleep(std::time::Duration::from_secs(1));
                continue;
            }
        };

        let work = match result.get("browserWork") {
            Some(w) if w.is_object() => w,
            _ => {
                eprintln!("⚠️ browserWork not found in template");
                std::thread::sleep(std::time::Duration::from_secs(1));
                continue;
            }
        };

        let height = work["height"].as_u64().unwrap_or(0);
        let candidate = work["candidate"].as_str().unwrap_or("").to_string();
        let header_hex = work["headerHex"].as_str().unwrap_or("").to_string();
        let target_hex = work["targetLEHex"].as_str().unwrap_or("").to_string();

        let header_bytes = match hex::decode(&header_hex) {
            Ok(b) if b.len() >= 76 => b,
            _ => {
                eprintln!("⚠️ Invalid headerHex length: {}", header_hex);
                continue;
            }
        };

        let target_bytes = match hex::decode(&target_hex) {
            Ok(b) if b.len() == 32 => b,
            _ => {
                eprintln!("⚠️ Invalid targetLEHex: {}", target_hex);
                continue;
            }
        };

        let midstate = compute_midstate(&header_bytes[0..64]);
        let mut header80 = [0u8; 80];
        header80[0..76].copy_from_slice(&header_bytes[0..76]);

        let mut target_arr = [0u8; 32];
        target_arr.copy_from_slice(&target_bytes[0..32]);

        let now_nano = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
        let mut start_nonce: u32 = (now_nano ^ (now_nano >> 32)) as u32;
        let block_start = Instant::now();

        // 3. Search nonces on GPU
        loop {
            let mut mining_res = CudaMiningResult {
                found: 0,
                nonce: 0,
                hash: [0u8; 32],
            };

            unsafe {
                cuda_miner_search(
                    midstate.as_ptr(),
                    header80.as_ptr(),
                    target_arr.as_ptr(),
                    start_nonce,
                    batch_size,
                    &mut mining_res,
                );
            }

            if mining_res.found != 0 {
                let winning_nonce = mining_res.nonce;
                let submit_res = rpc_call(
                    &rpc_url,
                    &cookie,
                    "submitblock",
                    serde_json::json!({
                        "browserCandidate": candidate,
                        "nonce": winning_nonce
                    }),
                );

                match submit_res {
                    Ok(val) => {
                        let res_obj = val.get("result");
                        let status = res_obj.and_then(|r| r.get("status")).and_then(|s| s.as_str()).unwrap_or("");
                        let hash = res_obj.and_then(|r| r.get("hash")).and_then(|s| s.as_str()).unwrap_or("");
                        if status == "accepted" {
                            total_blocks_found += 1;
                            let elapsed = block_start.elapsed().as_secs_f64();
                            let total_elapsed = global_start.elapsed().as_secs();
                            println!(
                                "🧱 [Height #{:03}/{} SOLVED!] In {:.2}s | Total Mined: {} | Elapsed: {:02}:{:02} | Nonce: {:#010x} | Hash: {}...",
                                height, target_height, elapsed, total_blocks_found, total_elapsed / 60, total_elapsed % 60, winning_nonce, &hash[..16]
                            );
                            break;
                        } else {
                            eprintln!("⚠️ Block #{} submit rejected: {:?}", height, val);
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("⚠️ submitblock RPC error: {}", e);
                        break;
                    }
                }
            }

            start_nonce = start_nonce.wrapping_add(batch_size);

            // Timeout after 15s to refresh block template if needed
            if block_start.elapsed().as_secs() > 15 {
                break;
            }
        }
    }

    unsafe { cuda_miner_cleanup() };
    println!("🏁 [Solo GPU Miner] Mining completed successfully!");
}
