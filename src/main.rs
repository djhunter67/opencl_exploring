use ocl::{
    core::{ClVersions, PlatformInfo},
    flags, Buffer, Context, Device, Kernel, Platform, Program, Queue,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let platforms = ocl::Platform::list();
    println!("\nNumber of platforms: {}", platforms.len());

    for platform in &platforms {
        println!("\nPlatform: {}", platform.name()?);
        Device::list_all(platform)?.iter().for_each(|device| {
            println!(
                "\t\tDevice:{}",
                device.name().expect("Failed to get device name")
            );
            println!(
                "\tPlatform available: {}",
                device.is_available().expect("Failed to get availability")
            );
            println!(
                "\tDevice vendor: {:?}",
                platform
                    .info(PlatformInfo::Vendor)
                    .expect("Failed to get vendor")
            );
            println!(
                "\tMax WG_SIZE: {}",
                device.max_wg_size().expect("Failed to get max wg size")
            );
            println!(
                "\tDevice version: {}",
                device.version().expect("Failed to get version")
            );
            println!(
                "\tDevice Version: {:?}",
                device
                    .device_versions()
                    .expect("Failed to get device versions")
            );
        });

        // for extend in platform.extensions()?.iter() {
        // println!("  Extension: {extend}");
        // }
    }

    trivial(platforms)?;

    Ok(())
}
fn trivial(platforms: Vec<Platform>) -> ocl::Result<()> {
    static KERNEL_SRC: &str = r"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }";

    // let src = include_str!("./assets/scaler_mul.cl");

    // (1) Define which platform and device(s) to use. Create a context,
    // queue, and program then define some dims (compare to step 1 above).

    // let platforms = Platform::list();

    // let platform = platforms
    // .into_iter()
    // .find(|p| p.name().expect("No name").contains("NVIDIA"))
    // .expect("No NVIDIA platform found");

    let platform = platforms
        .into_iter()
        .find(|p| p.name().expect("No name").contains("Intel"))
        .expect("No Intel platform found");

    println!("\n\nUsing platform: {}", platform.name()?);

    let list_all = Device::list_all(platform)?;
    let device = list_all.first().expect("No devices found");

    println!("Using device: {}", device.name()?);
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let program = Program::builder()
        .devices(device)
        .src(KERNEL_SRC)
        .build(&context)?;
    let queue = Queue::new(&context, *device, None)?;
    // let dims = 1 << 20; //1 << 20 = 2^20 = 1048576
    let dims = 1 << 7;
    // [NOTE]: At this point we could manually assemble a ProQue by calling:
    // `ProQue::new(context, queue, program, Some(dims))`. One might want to
    // do this when only one program and queue are all that's needed. Wrapping
    // it up into a single struct makes passing it around simpler.

    // (2) Create a `Buffer`:
    let buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(dims)
        .fill_val(0f32)
        .build()?;

    // (3) Create a kernel with arguments matching those in the source above:
    let kernel = Kernel::builder()
        .program(&program)
        .name("add")
        .queue(queue.clone())
        .global_work_size(dims)
        .arg(&buffer)
        .arg(10.0f32)
        .build()?;

    // (4) Run the kernel (default parameters shown for demonstration purposes):
    unsafe {
        kernel
            .cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(dims)
            .local_work_size(kernel.default_local_work_size())
            .enq()?;
    }

    // (5) Read results from the device into a vector (`::block` not shown):
    let mut vec = vec![0.0f32; dims];
    buffer.cmd().queue(&queue).offset(0).read(&mut vec).enq()?;

    // Print an element:
    println!("The value at index [{}] is now '{}'!", 27, vec[27]);
    Ok(())
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// fn matrix_ops() -> Result<(), ocl::Error> {							        //
//     Load the OpenCL kernel from a file							        //
//     let kernel_src =										        //
//         fs::read_to_string(path::Path::new("matrix_ops.cl")).expect("Failed to read kernel file");   //
// 												        //
//     Initialize OpenCL context								        //
//     let platforms = ocl::Platform::list().unwrap();						        //
//     let platform = platforms.first().unwrap();						        //
//     let devices = platform.devices().unwrap();						        //
//     let device = devices.first().unwrap();							        //
// 												        //
//     let ctx = ocl::Context::builder()							        //
//         .platform(platform)									        //
//         .devices(&[device])									        //
//         .build()										        //
//         .unwrap();										        //
// 												        //
//     Create command queue									        //
//     let queue = ocl::Queue::new(&ctx, &device).unwrap();					        //
// 												        //
//     Matrix size (for simplicity, using 2x2 matrices)						        //
//     const SIZE: i32 = 2;									        //
// 												        //
//     Example stiffness matrix K and damping matrix C for a simple system			        //
//     These matrices are placeholders; replace with real-world matrices as needed		        //
//     let k_matrix: Vec<f32> = vec![20.0, -10.0, -10.0, 20.0];					        //
// 												        //
//     let c_matrix: Vec<f32> = vec![5.0, -2.0, -2.0, 5.0];					        //
// 												        //
//     Initial guess for eigenvalue (lambda) and eigenvector (x)				        //
//     let mut lambda: f32 = 1.0;								        //
//     let mut x: Vec<f32> = vec![1.0, 1.0];							        //
// 												        //
//     OpenCL buffer creation									        //
//     let a_buffer = unsafe {									        //
//         ocl::Buffer::new(									        //
//             &ctx,										        //
//             ocl::MemoryFlags::empty(),							        //
//             k_matrix.len() as usize,								        //
//             None::<f32>,									        //
//         )											        //
//     }											        //
//     .unwrap();										        //
//     let x_buffer = ocl::Buffer::new(&ctx, ocl::MemoryFlags::empty(), x.len(), None::<f32>).unwrap(); //
//     let y_buffer = ocl::Buffer::new(&ctx, ocl::MemoryFlags::empty(), x.len(), None::<f32>).unwrap(); //
//     let residual_buffer =									        //
//         ocl::Buffer::new(&ctx, ocl::MemoryFlags::empty(), x.len(), None::<f32>).unwrap();	        //
// 												        //
//     Write data to buffers									        //
//     queue											        //
//         .write_buffer(&a_buffer, 0, &k_matrix)						        //
//         .enqueue()										        //
//         .unwrap();										        //
//     queue.write_buffer(&x_buffer, 0, &x).enqueue().unwrap();					        //
// 												        //
//     Build the OpenCL program									        //
//     let mut program = ocl::Program::builder()						        //
//         .src(kernel_src)									        //
//         .devices(&[device])									        //
//         .build(&ctx)										        //
//         .expect("Failed to build program");							        //
// 												        //
//     Create and set kernel arguments for matrix-vector multiplication				        //
//     let multiply_kernel = program.kernel("multiply_matrix_vector").unwrap();			        //
// 												        //
//     let _kernel_args = multiply_kernel							        //
//         .arg(&a_buffer)									        //
//         .arg(&x_buffer)									        //
//         .arg(&y_buffer)									        //
//         .arg(SIZE);										        //
// 												        //
//     Execute the kernel to compute Ax (matrix-vector product)					        //
//     let global_work_size = [SIZE as usize];							        //
//     multiply_kernel										        //
//         .enqueue(&queue, global_work_size, None, &())					        //
//         .unwrap();										        //
// 												        //
//     Read result from y_buffer								        //
//     let mut ax: Vec<f32> = vec![0.0; SIZE as usize];						        //
//     queue.read_buffer(&y_buffer, 0, &mut ax).enqueue().unwrap();				        //
// 												        //
//     Compute lambda*x (placeholder for actual eigenvalue iteration method)			        //
//     let mut lambda_x: Vec<f32> = x.iter().map(|val| lambda * val).collect();			        //
// 												        //
//     Create and set kernel arguments for residual calculation					        //
//     let residual_kernel = program.kernel("calculate_residual").unwrap();			        //
// 												        //
//     let _residual_args = residual_kernel							        //
//         .arg(&ax_buffer) Note: Need to write ax into buffer first				        //
//         .arg(&lambda_x_buffer)								        //
//         .arg(&residual_buffer)								        //
//         .arg(SIZE);										        //
// 												        //
//     Execute the kernel to compute residual (Ax - lambda*x)					        //
//     residual_kernel										        //
//         .enqueue(&queue, global_work_size, None, &())					        //
//         .unwrap();										        //
// 												        //
//     Read residual from buffer								        //
//     let mut residual: Vec<f32> = vec![0.0; SIZE as usize];					        //
//     queue											        //
//         .read_buffer(&residual_buffer, 0, &mut residual)					        //
//         .enqueue()										        //
//         .unwrap();										        //
// 												        //
//     Process and print results (placeholder for actual eigenvalue update)			        //
//     println!("Eigenvalue approximation: {}", lambda);					        //
//     println!("Residual vector: {:?}", residual);						        //
// 												        //
//     Cleanup buffers										        //
//     a_buffer.release();									        //
//     x_buffer.release();									        //
//     y_buffer.release();									        //
//     residual_buffer.release();								        //
// 												        //
//     Ok(())											        //
// }												        //
//////////////////////////////////////////////////////////////////////////////////////////////////////////
