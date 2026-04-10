use kajit_reprs::mir::{FunctionId, Graph, ProgramStorage, StorageError, parse_root_into_graph};

#[test]
fn mir_storage_resolves_entry_block_and_edges() {
    let source = r#"cfg_program vregs=1 slots=0 {
cfg_func @0 f0 entry=b0 {
data_args: []
data_results: []
block b0 params=[] insts=[1] term=t0 preds=[] succs=[1]
block b1 params=[] insts=[] term=t1 preds=[1] succs=[]
inst i1: copy
term t0: branch e1
term t1: return
edge e1: b0 -> b1 []
}
}"#;

    let mut graph = Graph::new();
    let handle = parse_root_into_graph(&mut graph, source).expect("MIR should parse");
    let program = graph.root(handle).expect("root should exist");
    let storage = ProgramStorage::new(&graph, program).expect("storage should build");
    let function = storage
        .function_storage(FunctionId::new(0))
        .expect("function storage should build")
        .expect("function should exist");

    let entry = function.entry_block().expect("entry block should resolve");
    assert_eq!(entry.id, kajit_reprs::mir::BlockId::new(0));

    let insts = function
        .block_insts(entry)
        .expect("block insts should resolve");
    assert_eq!(insts.len(), 1);
    assert_eq!(insts[0].id, kajit_reprs::mir::InstId::new(1));

    let term = function.block_term(entry).expect("term should resolve");
    let edges = function
        .terminator_edges(term)
        .expect("terminator edges should resolve");
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].id, kajit_reprs::mir::EdgeId::new(1));

    let target = function
        .edge_to(edges[0])
        .expect("edge target should resolve");
    assert_eq!(target.id, kajit_reprs::mir::BlockId::new(1));
}

#[test]
fn mir_storage_rejects_duplicate_block_ids() {
    let source = r#"cfg_program vregs=0 slots=0 {
cfg_func @0 f0 entry=b0 {
data_args: []
data_results: []
block b0 params=[] insts=[] term=t0 preds=[] succs=[]
block b0 params=[] insts=[] term=t1 preds=[] succs=[]
term t0: return
term t1: return
}
}"#;

    let mut graph = Graph::new();
    let handle = parse_root_into_graph(&mut graph, source).expect("MIR should parse");
    let program = graph.root(handle).expect("root should exist");
    let storage = ProgramStorage::new(&graph, program).expect("program storage should build");
    match storage.function_storage(FunctionId::new(0)) {
        Err(StorageError::DuplicateId {
            entity: "Block", ..
        }) => {}
        Err(other) => panic!("unexpected storage error: {other:?}"),
        Ok(_) => panic!("duplicate ids should fail"),
    }
}
