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

    trivial()?;

    Ok(())
}
fn trivial() -> ocl::Result<()> {
    static KERNEL_SRC: &str = r"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }";

    // let src = include_str!("./assets/scaler_mul.cl");

    // (1) Define which platform and device(s) to use. Create a context,
    // queue, and program then define some dims (compare to step 1 above).

    let platforms = Platform::list();

    println!(
        "\n\nAll platforms:\n{}",
        platforms
            .iter()
            .map(|p| p.name().expect("Failed to get platform name"))
            .collect::<Vec<_>>()
            .join("\n")
    );

    let platform = platforms
        .into_iter()
        .find(|p| p.name().expect("No name").contains("NVIDIA"))
        .expect("No NVIDIA platform found");

    // let platform = Platform::list()
    //     .into_iter()
    //     .find(|p| p.name().expect("No name").contains("Intel"))
    //     .expect("No Intel platform found");

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
