#![feature(lang_items)]
#![no_std]
#![feature(register_attr)]
#![register_attr(spirv)]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[lang = "eh_personality"]
extern "C" fn rust_eh_personality() {}

#[spirv(vertex)]
pub fn main() {
}