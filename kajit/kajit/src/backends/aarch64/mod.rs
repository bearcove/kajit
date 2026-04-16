use std::collections::BTreeMap;

use crate::regalloc_engine::ir;

pub mod regalloc3_backend;

pub(crate) fn build_debug_line_maps(
    program: &ir::Program,
) -> (BTreeMap<(u32, ir::OpId), u32>, BTreeMap<u32, u32>) {
    let mut line_by_lambda_op = BTreeMap::<(u32, ir::OpId), u32>::new();
    let mut first_line_by_lambda = BTreeMap::<u32, u32>::new();
    let mut next_line = 1u32;
    for func in &program.funcs {
        let lambda_id = func.lambda_id.index() as u32;
        let mut first_line = None::<u32>;
        for block in func.live_blocks() {
            for inst_id in &block.insts {
                let op_id = ir::OpId::Inst(*inst_id);
                line_by_lambda_op.insert((lambda_id, op_id), next_line);
                if first_line.is_none() {
                    first_line = Some(next_line);
                }
                next_line += 1;
            }
            let term_op = ir::OpId::Term(block.term);
            line_by_lambda_op.insert((lambda_id, term_op), next_line);
            if first_line.is_none() {
                first_line = Some(next_line);
            }
            next_line += 1;
        }
        first_line_by_lambda.insert(lambda_id, first_line.unwrap_or(1));
    }
    (line_by_lambda_op, first_line_by_lambda)
}
