pub fn normalize_inst(text: &str) -> String {
    if let Some(rest) = text.strip_prefix("mov ")
        && let Some((reg, imm)) = rest.split_once(", 0x")
    {
        let hex_len = imm.len();
        if reg.starts_with('r') && hex_len >= 10 {
            return format!("mov {reg}, 0x<imm>");
        }
    }

    if let Some(rest) = text.strip_prefix("adrp ")
        && let Some((reg, imm)) = rest.split_once(", 0x")
    {
        let hex_len = imm.len();
        if reg.starts_with('x') && hex_len >= 6 {
            return format!("adrp {reg}, 0x<imm>");
        }
    }

    for op in ["mov", "movk"] {
        let op_prefix = format!("{op} ");
        if let Some(rest) = text.strip_prefix(&op_prefix)
            && let Some((reg, imm_tail)) = rest.split_once(", #")
        {
            if !reg.starts_with('x') {
                continue;
            }
            if let Some((_, suffix)) = imm_tail.split_once(',') {
                return format!("{op_prefix}{reg}, #<imm>,{suffix}");
            }
            return format!("{op_prefix}{reg}, #<imm>");
        }
    }

    text.to_owned()
}
