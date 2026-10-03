#![allow(dead_code)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::sync::OnceLock;

pub type ClInt = isize;
pub type ClUint = u32;
pub type ClUlong = u64;

pub const CL_DEVICE_TYPE_GPU: ClUlong = 1 << 2;
pub const CL_DEVICE_NAME: ClUint = 0x102B;
pub const CL_DEVICE_VENDOR: ClUint = 0x102C;
pub const CL_DEVICE_GLOBAL_MEM_SIZE: ClUint = 0x101F;
pub const CL_DEVICE_MAX_COMPUTE_UNITS: ClUint = 0x1002;
pub const CL_PLATFORM_NAME: ClUint = 0x0902;
pub const CL_MEM_READ_WRITE: ClUlong = 1 << 0;
pub const CL_MEM_WRITE_ONLY: ClUlong = 1 << 1;
pub const CL_MEM_READ_ONLY: ClUlong = 1 << 2;
pub const CL_MEM_COPY_HOST_PTR: ClUlong = 1 << 5;
pub const CL_PROGRAM_BUILD_LOG: ClUint = 0x1114;
pub const CL_TRUE: ClUint = 1;

type PfnClGetPlatformIDs = unsafe extern "C" fn(ClUint, *mut *mut c_void, *mut ClUint) -> ClInt;
type PfnClGetPlatformInfo = unsafe extern "C" fn(*mut c_void, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
type PfnClGetDeviceIDs = unsafe extern "C" fn(*mut c_void, ClUlong, ClUint, *mut *mut c_void, *mut ClUint) -> ClInt;
type PfnClGetDeviceInfo = unsafe extern "C" fn(*mut c_void, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
type PfnClCreateContext = unsafe extern "C" fn(*const c_void, ClUint, *const *mut c_void, *const c_void, *const c_void, *mut ClInt) -> *mut c_void;
type PfnClCreateCommandQueue = unsafe extern "C" fn(*mut c_void, *mut c_void, ClUlong, *mut ClInt) -> *mut c_void;
type PfnClCreateProgramWithSource = unsafe extern "C" fn(*mut c_void, ClUint, *const *const c_char, *const usize, *mut ClInt) -> *mut c_void;
type PfnClBuildProgram = unsafe extern "C" fn(*mut c_void, ClUint, *const *mut c_void, *const c_char, *const c_void, *const c_void) -> ClInt;
type PfnClGetProgramBuildInfo = unsafe extern "C" fn(*mut c_void, *mut c_void, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
type PfnClCreateKernel = unsafe extern "C" fn(*mut c_void, *const c_char, *mut ClInt) -> *mut c_void;
type PfnClCreateBuffer = unsafe extern "C" fn(*mut c_void, ClUlong, usize, *mut c_void, *mut ClInt) -> *mut c_void;
type PfnClSetKernelArg = unsafe extern "C" fn(*mut c_void, ClUint, usize, *const c_void) -> ClInt;
type PfnClEnqueueNDRangeKernel = unsafe extern "C" fn(*mut c_void, *mut c_void, ClUint, *const usize, *const usize, *const usize, ClUint, *const c_void, *mut c_void) -> ClInt;
type PfnClEnqueueWriteBuffer = unsafe extern "C" fn(*mut c_void, *mut c_void, ClUint, usize, usize, *const c_void, ClUint, *const c_void, *mut c_void) -> ClInt;
type PfnClEnqueueReadBuffer = unsafe extern "C" fn(*mut c_void, *mut c_void, ClUint, usize, usize, *mut c_void, ClUint, *const c_void, *mut c_void) -> ClInt;
type PfnClFinish = unsafe extern "C" fn(*mut c_void) -> ClInt;
type PfnClReleaseMemObject = unsafe extern "C" fn(*mut c_void) -> ClInt;
type PfnClReleaseKernel = unsafe extern "C" fn(*mut c_void) -> ClInt;
type PfnClReleaseProgram = unsafe extern "C" fn(*mut c_void) -> ClInt;
type PfnClReleaseCommandQueue = unsafe extern "C" fn(*mut c_void) -> ClInt;
type PfnClReleaseContext = unsafe extern "C" fn(*mut c_void) -> ClInt;

#[derive(Clone)]
pub struct OpenClLib {
    cl_get_platform_ids: PfnClGetPlatformIDs,
    cl_get_platform_info: PfnClGetPlatformInfo,
    cl_get_device_ids: PfnClGetDeviceIDs,
    cl_get_device_info: PfnClGetDeviceInfo,
    cl_create_context: PfnClCreateContext,
    cl_create_command_queue: PfnClCreateCommandQueue,
    cl_create_program_with_source: PfnClCreateProgramWithSource,
    cl_build_program: PfnClBuildProgram,
    cl_get_program_build_info: PfnClGetProgramBuildInfo,
    cl_create_kernel: PfnClCreateKernel,
    cl_create_buffer: PfnClCreateBuffer,
    cl_set_kernel_arg: PfnClSetKernelArg,
    cl_enqueue_nd_range_kernel: PfnClEnqueueNDRangeKernel,
    cl_enqueue_write_buffer: PfnClEnqueueWriteBuffer,
    cl_enqueue_read_buffer: PfnClEnqueueReadBuffer,
    cl_finish: PfnClFinish,
    cl_release_mem_object: PfnClReleaseMemObject,
    cl_release_kernel: PfnClReleaseKernel,
    cl_release_program: PfnClReleaseProgram,
    cl_release_command_queue: PfnClReleaseCommandQueue,
    cl_release_context: PfnClReleaseContext,
}

unsafe impl Send for OpenClLib {}
unsafe impl Sync for OpenClLib {}

static OPENCL_LIB: OnceLock<Option<OpenClLib>> = OnceLock::new();

#[cfg(target_os = "windows")]
extern "system" {
    fn LoadLibraryA(lpLibFileName: *const c_char) -> *mut c_void;
    fn GetProcAddress(hModule: *mut c_void, lpProcName: *const c_char) -> *mut c_void;
}

pub fn get_opencl_lib() -> Option<&'static OpenClLib> {
    OPENCL_LIB.get_or_init(|| {
        #[cfg(target_os = "windows")]
        let lib_names = ["OpenCL.dll"];
        #[cfg(target_os = "macos")]
        let lib_names = ["/System/Library/Frameworks/OpenCL.framework/OpenCL", "libOpenCL.dylib"];
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        let lib_names = ["libOpenCL.so.1", "libOpenCL.so", "libnvidia-opencl.so.1"];

        for &name in &lib_names {
            let c_name = CString::new(name).unwrap();

            #[cfg(target_os = "windows")]
            let handle = unsafe { LoadLibraryA(c_name.as_ptr()) };

            #[cfg(not(target_os = "windows"))]
            let handle = unsafe { libc::dlopen(c_name.as_ptr(), libc::RTLD_NOW) };

            if !handle.is_null() {
                unsafe {
                    macro_rules! load_sym {
                        ($sym:ident, $type:ty) => {{
                            let s_name = CString::new(stringify!($sym)).unwrap();
                            #[cfg(target_os = "windows")]
                            let p = GetProcAddress(handle, s_name.as_ptr());
                            #[cfg(not(target_os = "windows"))]
                            let p = libc::dlsym(handle, s_name.as_ptr());
                            if p.is_null() {
                                return None;
                            }
                            std::mem::transmute::<*mut c_void, $type>(p)
                        }};
                    }

                    return Some(OpenClLib {
                        cl_get_platform_ids: load_sym!(clGetPlatformIDs, PfnClGetPlatformIDs),
                        cl_get_platform_info: load_sym!(clGetPlatformInfo, PfnClGetPlatformInfo),
                        cl_get_device_ids: load_sym!(clGetDeviceIDs, PfnClGetDeviceIDs),
                        cl_get_device_info: load_sym!(clGetDeviceInfo, PfnClGetDeviceInfo),
                        cl_create_context: load_sym!(clCreateContext, PfnClCreateContext),
                        cl_create_command_queue: load_sym!(clCreateCommandQueue, PfnClCreateCommandQueue),
                        cl_create_program_with_source: load_sym!(clCreateProgramWithSource, PfnClCreateProgramWithSource),
                        cl_build_program: load_sym!(clBuildProgram, PfnClBuildProgram),
                        cl_get_program_build_info: load_sym!(clGetProgramBuildInfo, PfnClGetProgramBuildInfo),
                        cl_create_kernel: load_sym!(clCreateKernel, PfnClCreateKernel),
                        cl_create_buffer: load_sym!(clCreateBuffer, PfnClCreateBuffer),
                        cl_set_kernel_arg: load_sym!(clSetKernelArg, PfnClSetKernelArg),
                        cl_enqueue_nd_range_kernel: load_sym!(clEnqueueNDRangeKernel, PfnClEnqueueNDRangeKernel),
                        cl_enqueue_write_buffer: load_sym!(clEnqueueWriteBuffer, PfnClEnqueueWriteBuffer),
                        cl_enqueue_read_buffer: load_sym!(clEnqueueReadBuffer, PfnClEnqueueReadBuffer),
                        cl_finish: load_sym!(clFinish, PfnClFinish),
                        cl_release_mem_object: load_sym!(clReleaseMemObject, PfnClReleaseMemObject),
                        cl_release_kernel: load_sym!(clReleaseKernel, PfnClReleaseKernel),
                        cl_release_program: load_sym!(clReleaseProgram, PfnClReleaseProgram),
                        cl_release_command_queue: load_sym!(clReleaseCommandQueue, PfnClReleaseCommandQueue),
                        cl_release_context: load_sym!(clReleaseContext, PfnClReleaseContext),
                    });
                }
            }
        }
        None
    }).as_ref()
}

#[derive(Clone, Debug)]
pub struct OpenClDeviceInfo {
    pub platform_idx: usize,
    pub device_idx: usize,
    pub platform_id: usize,
    pub device_id: usize,
    pub platform_name: String,
    pub device_name: String,
    pub vendor: String,
    pub is_nvidia: bool,
    pub total_mem_bytes: usize,
    pub compute_units: u32,
}

pub fn get_opencl_devices() -> Vec<OpenClDeviceInfo> {
    let cl = match get_opencl_lib() {
        Some(lib) => lib,
        None => return Vec::new(),
    };

    let mut devices_list = Vec::new();
    unsafe {
        let mut num_platforms: ClUint = 0;
        if (cl.cl_get_platform_ids)(0, std::ptr::null_mut(), &mut num_platforms) != 0 || num_platforms == 0 {
            return devices_list;
        }

        let mut platforms = vec![std::ptr::null_mut(); num_platforms as usize];
        (cl.cl_get_platform_ids)(num_platforms, platforms.as_mut_ptr(), std::ptr::null_mut());

        for (p_idx, &p) in platforms.iter().enumerate() {
            let mut p_name_buf = [0 as c_char; 256];
            (cl.cl_get_platform_info)(p, CL_PLATFORM_NAME, p_name_buf.len(), p_name_buf.as_mut_ptr() as *mut c_void, std::ptr::null_mut());
            let p_name = CStr::from_ptr(p_name_buf.as_ptr()).to_string_lossy().trim().to_string();

            let mut num_devices: ClUint = 0;
            if (cl.cl_get_device_ids)(p, CL_DEVICE_TYPE_GPU, 0, std::ptr::null_mut(), &mut num_devices) == 0 && num_devices > 0 {
                let mut devs = vec![std::ptr::null_mut(); num_devices as usize];
                (cl.cl_get_device_ids)(p, CL_DEVICE_TYPE_GPU, num_devices, devs.as_mut_ptr(), std::ptr::null_mut());

                for (d_idx, &d) in devs.iter().enumerate() {
                    let mut d_name_buf = [0 as c_char; 256];
                    (cl.cl_get_device_info)(d, CL_DEVICE_NAME, d_name_buf.len(), d_name_buf.as_mut_ptr() as *mut c_void, std::ptr::null_mut());
                    let d_name = CStr::from_ptr(d_name_buf.as_ptr()).to_string_lossy().trim().to_string();

                    let mut d_vendor_buf = [0 as c_char; 256];
                    (cl.cl_get_device_info)(d, CL_DEVICE_VENDOR, d_vendor_buf.len(), d_vendor_buf.as_mut_ptr() as *mut c_void, std::ptr::null_mut());
                    let d_vendor = CStr::from_ptr(d_vendor_buf.as_ptr()).to_string_lossy().trim().to_string();

                    let mut mem_size: ClUlong = 0;
                    (cl.cl_get_device_info)(d, CL_DEVICE_GLOBAL_MEM_SIZE, 8, &mut mem_size as *mut _ as *mut c_void, std::ptr::null_mut());

                    let mut cu: ClUint = 0;
                    (cl.cl_get_device_info)(d, CL_DEVICE_MAX_COMPUTE_UNITS, 4, &mut cu as *mut _ as *mut c_void, std::ptr::null_mut());

                    let is_nvidia = p_name.to_lowercase().contains("nvidia")
                        || d_name.to_lowercase().contains("nvidia")
                        || d_vendor.to_lowercase().contains("nvidia");

                    devices_list.push(OpenClDeviceInfo {
                        platform_idx: p_idx,
                        device_idx: d_idx,
                        platform_id: p as usize,
                        device_id: d as usize,
                        platform_name: p_name.clone(),
                        device_name: d_name,
                        vendor: d_vendor,
                        is_nvidia,
                        total_mem_bytes: mem_size as usize,
                        compute_units: cu,
                    });
                }
            }
        }
    }
    devices_list
}

pub static OPENCL_SOURCE: &str = include_str!("tru_opencl.cl");

pub struct OpenClWorker {
    cl: OpenClLib,
    context: *mut c_void,
    queue: *mut c_void,
    program: *mut c_void,
    kernel: *mut c_void,

    d_midstate: *mut c_void,
    d_target: *mut c_void,
    d_found: *mut c_void,
    d_found_nonce: *mut c_void,
    d_found_hash: *mut c_void,

    last_midstate: [u32; 8],
    last_target: [u8; 32],
    initialized: bool,
}

unsafe impl Send for OpenClWorker {}
unsafe impl Sync for OpenClWorker {}

impl OpenClWorker {
    pub fn new(device_info: &OpenClDeviceInfo) -> Result<Self, String> {
        let cl = get_opencl_lib().ok_or_else(|| "OpenCL library is not available".to_string())?.clone();
        let device = device_info.device_id as *mut c_void;

        unsafe {
            let mut err: ClInt = 0;
            let context = (cl.cl_create_context)(std::ptr::null(), 1, &device, std::ptr::null(), std::ptr::null(), &mut err);
            if err != 0 || context.is_null() {
                return Err(format!("clCreateContext failed: {}", err));
            }

            let queue = (cl.cl_create_command_queue)(context, device, 0, &mut err);
            if err != 0 || queue.is_null() {
                (cl.cl_release_context)(context);
                return Err(format!("clCreateCommandQueue failed: {}", err));
            }

            let src_c = CString::new(OPENCL_SOURCE).unwrap();
            let src_ptr = src_c.as_ptr();
            let src_len = src_c.as_bytes().len();

            let program = (cl.cl_create_program_with_source)(context, 1, &src_ptr, &src_len, &mut err);
            if err != 0 || program.is_null() {
                (cl.cl_release_command_queue)(queue);
                (cl.cl_release_context)(context);
                return Err(format!("clCreateProgramWithSource failed: {}", err));
            }

            let build_err = (cl.cl_build_program)(program, 1, &device, std::ptr::null(), std::ptr::null(), std::ptr::null());
            if build_err != 0 {
                let mut log_len: usize = 0;
                (cl.cl_get_program_build_info)(program, device, CL_PROGRAM_BUILD_LOG, 0, std::ptr::null_mut(), &mut log_len);
                let mut log = vec![0u8; log_len];
                (cl.cl_get_program_build_info)(program, device, CL_PROGRAM_BUILD_LOG, log_len, log.as_mut_ptr() as *mut c_void, std::ptr::null_mut());
                let log_str = String::from_utf8_lossy(&log).into_owned();
                (cl.cl_release_program)(program);
                (cl.cl_release_command_queue)(queue);
                (cl.cl_release_context)(context);
                return Err(format!("clBuildProgram failed: {}\nLog: {}", build_err, log_str));
            }

            let k_name = CString::new("tru_opencl_mine").unwrap();
            let kernel = (cl.cl_create_kernel)(program, k_name.as_ptr(), &mut err);
            if err != 0 || kernel.is_null() {
                (cl.cl_release_program)(program);
                (cl.cl_release_command_queue)(queue);
                (cl.cl_release_context)(context);
                return Err(format!("clCreateKernel tru_opencl_mine failed: {}", err));
            }

            // Allocate device buffers
            let d_midstate = (cl.cl_create_buffer)(context, CL_MEM_READ_ONLY, 32, std::ptr::null_mut(), &mut err);
            let d_target = (cl.cl_create_buffer)(context, CL_MEM_READ_ONLY, 32, std::ptr::null_mut(), &mut err);
            let d_found = (cl.cl_create_buffer)(context, CL_MEM_READ_WRITE, 4, std::ptr::null_mut(), &mut err);
            let d_found_nonce = (cl.cl_create_buffer)(context, CL_MEM_WRITE_ONLY, 4, std::ptr::null_mut(), &mut err);
            let d_found_hash = (cl.cl_create_buffer)(context, CL_MEM_WRITE_ONLY, 32, std::ptr::null_mut(), &mut err);

            if d_midstate.is_null() || d_target.is_null() || d_found.is_null() || d_found_nonce.is_null() || d_found_hash.is_null() {
                return Err("Failed to allocate OpenCL GPU buffers".to_string());
            }

            Ok(Self {
                cl,
                context,
                queue,
                program,
                kernel,
                d_midstate,
                d_target,
                d_found,
                d_found_nonce,
                d_found_hash,
                last_midstate: [0u32; 8],
                last_target: [0u8; 32],
                initialized: false,
            })
        }
    }

    pub fn search(
        &mut self,
        midstate: &[u32; 8],
        h80: &[u8; 80],
        target: &[u8; 32],
        start_nonce: u32,
        batch_size: u32,
    ) -> Result<Option<(u32, [u8; 32])>, String> {
        let w0 = u32::from_be_bytes([h80[64], h80[65], h80[66], h80[67]]);
        let w1 = u32::from_be_bytes([h80[68], h80[69], h80[70], h80[71]]);
        let w2 = u32::from_be_bytes([h80[72], h80[73], h80[74], h80[75]]);

        unsafe {
            let zero: i32 = 0;
            // Upload midstate & target if changed
            if !self.initialized || self.last_midstate != *midstate {
                (self.cl.cl_enqueue_write_buffer)(self.queue, self.d_midstate, CL_TRUE, 0, 32, midstate.as_ptr() as *const c_void, 0, std::ptr::null(), std::ptr::null_mut());
                self.last_midstate = *midstate;
            }
            if !self.initialized || self.last_target != *target {
                (self.cl.cl_enqueue_write_buffer)(self.queue, self.d_target, CL_TRUE, 0, 32, target.as_ptr() as *const c_void, 0, std::ptr::null(), std::ptr::null_mut());
                self.last_target = *target;
            }
            self.initialized = true;

            // Reset found flag
            (self.cl.cl_enqueue_write_buffer)(self.queue, self.d_found, CL_TRUE, 0, 4, &zero as *const _ as *const c_void, 0, std::ptr::null(), std::ptr::null_mut());

            // Set kernel arguments
            (self.cl.cl_set_kernel_arg)(self.kernel, 0, std::mem::size_of::<*mut c_void>(), &self.d_midstate as *const _ as *const c_void);
            (self.cl.cl_set_kernel_arg)(self.kernel, 1, 4, &w0 as *const _ as *const c_void);
            (self.cl.cl_set_kernel_arg)(self.kernel, 2, 4, &w1 as *const _ as *const c_void);
            (self.cl.cl_set_kernel_arg)(self.kernel, 3, 4, &w2 as *const _ as *const c_void);
            (self.cl.cl_set_kernel_arg)(self.kernel, 4, 4, &start_nonce as *const _ as *const c_void);
            (self.cl.cl_set_kernel_arg)(self.kernel, 5, 4, &batch_size as *const _ as *const c_void);
            (self.cl.cl_set_kernel_arg)(self.kernel, 6, std::mem::size_of::<*mut c_void>(), &self.d_target as *const _ as *const c_void);
            (self.cl.cl_set_kernel_arg)(self.kernel, 7, std::mem::size_of::<*mut c_void>(), &self.d_found as *const _ as *const c_void);
            (self.cl.cl_set_kernel_arg)(self.kernel, 8, std::mem::size_of::<*mut c_void>(), &self.d_found_nonce as *const _ as *const c_void);
            (self.cl.cl_set_kernel_arg)(self.kernel, 9, std::mem::size_of::<*mut c_void>(), &self.d_found_hash as *const _ as *const c_void);

            let gws = batch_size as usize;
            let lws: usize = 256;
            let err = (self.cl.cl_enqueue_nd_range_kernel)(self.queue, self.kernel, 1, std::ptr::null(), &gws, &lws, 0, std::ptr::null(), std::ptr::null_mut());
            if err != 0 {
                return Err(format!("clEnqueueNDRangeKernel failed: {}", err));
            }

            let mut found: i32 = 0;
            (self.cl.cl_enqueue_read_buffer)(self.queue, self.d_found, CL_TRUE, 0, 4, &mut found as *mut _ as *mut c_void, 0, std::ptr::null(), std::ptr::null_mut());

            if found != 0 {
                let mut nonce: u32 = 0;
                let mut hash = [0u8; 32];
                (self.cl.cl_enqueue_read_buffer)(self.queue, self.d_found_nonce, CL_TRUE, 0, 4, &mut nonce as *mut _ as *mut c_void, 0, std::ptr::null(), std::ptr::null_mut());
                (self.cl.cl_enqueue_read_buffer)(self.queue, self.d_found_hash, CL_TRUE, 0, 32, hash.as_mut_ptr() as *mut c_void, 0, std::ptr::null(), std::ptr::null_mut());
                return Ok(Some((nonce, hash)));
            }

            Ok(None)
        }
    }
}

impl Drop for OpenClWorker {
    fn drop(&mut self) {
        unsafe {
            if !self.d_midstate.is_null() { (self.cl.cl_release_mem_object)(self.d_midstate); }
            if !self.d_target.is_null() { (self.cl.cl_release_mem_object)(self.d_target); }
            if !self.d_found.is_null() { (self.cl.cl_release_mem_object)(self.d_found); }
            if !self.d_found_nonce.is_null() { (self.cl.cl_release_mem_object)(self.d_found_nonce); }
            if !self.d_found_hash.is_null() { (self.cl.cl_release_mem_object)(self.d_found_hash); }
            if !self.kernel.is_null() { (self.cl.cl_release_kernel)(self.kernel); }
            if !self.program.is_null() { (self.cl.cl_release_program)(self.program); }
            if !self.queue.is_null() { (self.cl.cl_release_command_queue)(self.queue); }
            if !self.context.is_null() { (self.cl.cl_release_context)(self.context); }
        }
    }
}
