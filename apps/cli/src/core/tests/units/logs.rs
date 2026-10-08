//! Log buffers and the files unit output is written to.

use crate::core::types::{LogLine, Stream};
use crate::units::logs::{LogBuffer, LogWriter};

#[test]
fn log_buffer_drops_oldest_and_searches_wrapping() {
    let mut b = LogBuffer::new(3);
    for t in ["alpha", "Beta", "gamma", "delta"] {
        b.push(LogLine::now(Stream::Out, t));
    }
    let texts: Vec<&str> = b.lines().map(|l| l.text.as_str()).collect();
    assert_eq!(texts, ["Beta", "gamma", "delta"]);
    assert_eq!(b.find("beta", 2, false), Some(0));
    assert_eq!(b.find("delta", 0, true), Some(2));
    assert_eq!(b.find("zeta", 0, false), None);
    assert_eq!(b.find("", 0, false), None);
}

#[test]
fn log_writer_persists_unit_output_under_its_directory() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("sparky-cli-logs-{}", uuid::Uuid::new_v4()));
    let mut writer = LogWriter::new(&dir, 0)?;

    writer.append("engine", &LogLine::now(Stream::Out, "ready"))?;

    let saved = std::fs::read_to_string(dir.join("engine.log"))?;
    assert!(saved.ends_with(" out ready\n"));
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn a_unit_log_past_its_size_is_rotated_to_one_old_file() -> std::io::Result<()> {
    let dir = std::env::temp_dir().join(format!("sparky-rotate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut writer = LogWriter::new(&dir, 200)?;
    for i in 0..40 {
        writer.append(
            "engine",
            &LogLine::now(Stream::Out, format!("line {i} {}", "x".repeat(20))),
        )?;
    }
    let current = std::fs::metadata(dir.join("engine.log"))?.len();
    let old = std::fs::metadata(dir.join("engine.log.1"))?.len();
    assert!(
        current <= 260,
        "the live file stays near the cap: {current}"
    );
    assert!(old <= 260, "one old file, also near the cap: {old}");
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
