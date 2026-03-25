use lldb_sys as lldb;

fn main() {
    unsafe {
        lldb::SBDebuggerInitialize();
        let dbg = lldb::SBDebuggerCreate();
        println!("{:?}", dbg);
        lldb::SBDebuggerTerminate();
    }
}
