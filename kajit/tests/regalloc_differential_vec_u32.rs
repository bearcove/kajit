use facet::Facet;
use kajit_mir::DifferentialCheckResult;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
struct Nums {
    vals: Vec<u32>,
}

#[test]
fn postcard_vec_u32_v0_regalloc_differential_matches() {
    let value = Nums {
        vals: vec![1, 2, 3],
    };
    let input = postcard::to_allocvec(&value).expect("serialize postcard input");
    let program = kajit::debug_ra_program(Nums::SHAPE, &kajit::postcard::KajitPostcard);
    let lambda_ids: Vec<usize> = program.funcs.iter().map(|f| f.lambda_id.index()).collect();
    assert!(
        program.funcs.len() > 1,
        "expected lambda body functions in RA program; ids={lambda_ids:?}"
    );
    assert!(
        lambda_ids.contains(&1),
        "expected lambda id @1 in RA program; ids={lambda_ids:?}"
    );
    let result = kajit_mir::differential_check_program(program, &input);
    match result {
        DifferentialCheckResult::Match { .. } => {}
        other => panic!("regalloc differential mismatch for postcard vec<u32>: {other:?}"),
    }
}

#[test]
fn postcard_vec_u32_v0_linear_49_has_term_cond_alloc() {
    let value = Nums {
        vals: vec![1, 2, 3],
    };
    let _input = postcard::to_allocvec(&value).expect("serialize postcard input");
    let program = kajit::debug_ra_program(Nums::SHAPE, &kajit::postcard::KajitPostcard);
    let allocated = kajit::regalloc_engine::allocate_program(&program).expect("allocate program");
    let lambda1 = allocated
        .functions
        .iter()
        .find(|f| f.lambda_id.index() == 1)
        .expect("lambda @1 alloc should exist");
    let term_indices: HashSet<usize> = lambda1
        .term_inst_indices_by_block
        .iter()
        .flatten()
        .copied()
        .collect();

    let mut chosen_len = None::<usize>;
    let mut raw = Vec::new();
    for (inst_index, maybe_linear) in lambda1.inst_linear_op_indices.iter().copied().enumerate() {
        if maybe_linear != Some(49) {
            continue;
        }
        let allocs_len = lambda1
            .inst_allocs
            .get(inst_index)
            .map(|allocs| allocs.len())
            .unwrap_or(0);
        let is_term = term_indices.contains(&inst_index);
        raw.push((inst_index, is_term, allocs_len));

        match &mut chosen_len {
            None => chosen_len = Some(allocs_len),
            Some(existing_len) => {
                if !is_term || *existing_len < allocs_len {
                    *existing_len = allocs_len;
                }
            }
        }
    }

    let chosen_len = chosen_len.unwrap_or(0);
    assert!(
        chosen_len >= 1,
        "lambda @1 linear 49 should expose at least one alloc for branch cond; raw={raw:?}, chosen_len={chosen_len}"
    );
}
