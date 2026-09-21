//! Files the caller attached, copied into the sandbox workspace before the first model call.

use super::Agent;
use super::execute::workspace_session;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::file::FileAttachment;
use crate::core::types::tools::sandbox::SandboxRequest;
use crate::core::types::trace::TraceEvent;

/// Words too common to find a passage by.
const COMMON: [&str; 40] = [
    "the", "and", "for", "are", "was", "were", "what", "when", "where", "which", "who", "why",
    "how", "does", "did", "this", "that", "these", "those", "with", "from", "into", "about",
    "have", "has", "had", "you", "your", "can", "could", "would", "should", "tell", "please",
    "file", "form", "document", "attached", "there", "they",
];

/// The words of question a passage is ranked by: lowercase, three letters or more, not common,
/// each once, at most twelve. Letters and digits only, so they are safe on a command line.
pub fn terms(question: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for word in question
        .to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
    {
        if word.len() >= 3 && !COMMON.contains(&word) && !out.iter().any(|w| w == word) {
            out.push(word.to_owned());
        }
        if out.len() == 12 {
            break;
        }
    }
    out
}

/// The sandbox command that turns an uploaded file at path into text beside it at path.txt, then
/// prints the character count, the first preview characters, and the passages holding the most
/// of terms, up to matches characters.
///
/// A PDF is read by pdftotext, and by tesseract over its first ocr_pages pages when it holds no
/// text layer. Word, Excel and CSV go through python3. Any other text file is copied; a binary
/// file gives empty text.
pub fn to_text(
    path: &str,
    terms: &[String],
    preview: usize,
    matches: usize,
    ocr_pages: u32,
) -> String {
    let words = terms.join(" ");
    format!(
        r#"f={path}; t={path}.txt; : > "$t"
case "$(printf %s "$f" | tr A-Z a-z)" in
*.pdf) pdftotext -layout "$f" "$t" 2>/dev/null
  if [ "$(tr -d '[:space:]' < "$t" | wc -c)" -lt 10 ]; then
    pdftoppm -r 150 -png -l {ocr_pages} "$f" /tmp/ocr-page 2>/dev/null
    for p in /tmp/ocr-page*.png; do [ -e "$p" ] && tesseract "$p" - 2>/dev/null; done > "$t"
    rm -f /tmp/ocr-page*.png; echo OCR
  fi ;;
*.docx) python3 -c 'import docx,sys
d=docx.Document(sys.argv[1])
print("\n".join(p.text for p in d.paragraphs))
for tb in d.tables:
    for r in tb.rows: print("\t".join(c.text for c in r.cells))' "$f" > "$t" 2>/dev/null ;;
*.xlsx|*.xls) python3 -c 'import pandas,sys
for n,df in pandas.read_excel(sys.argv[1],sheet_name=None).items(): print("sheet: "+str(n)); print(df.to_csv(index=False))' "$f" > "$t" 2>/dev/null ;;
*) file -b --mime-type "$f" | grep -q -e '^text/' -e json -e xml && cp "$f" "$t" ;;
esac
echo "CHARS $(wc -c < "$t")"
echo "=== PREVIEW"
head -c {preview} "$t"
echo
echo "=== MATCHES"
python3 -c 'import sys
lines=open(sys.argv[1],errors="replace").read().splitlines()
terms=sys.argv[3:]; limit=int(sys.argv[2])
ranked=[]
for i in range(len(lines)):
    window=" ".join(lines[max(0,i-1):i+3]).lower()
    score=sum(1 for w in terms if w in window)
    if score: ranked.append((-score,i))
ranked.sort()
kept=[]; used=0
for _,i in ranked:
    if any(abs(i-j)<5 for j in kept): continue
    block="\n".join(l.rstrip() for l in lines[max(0,i-1):i+5]).strip()
    if used+len(block)>limit: continue
    kept.append(i); used+=len(block)
    print("[line "+str(i+1)+"]\n"+block+"\n")
    if len(kept)==8: break' "$t" {matches} {words}"#
    )
}

/// What the conversion printed: whether it took OCR, the character count, the preview, and the
/// passages that match the question.
pub fn converted(stdout: &str) -> Converted {
    let ocr = stdout.lines().next().is_some_and(|l| l.trim() == "OCR");
    let chars = stdout
        .lines()
        .find_map(|l| l.trim().strip_prefix("CHARS "))
        .and_then(|n| n.trim().parse().ok())
        .unwrap_or(0);
    let after = |marker: &str| stdout.split_once(marker).map_or("", |(_, rest)| rest);
    let preview = after("=== PREVIEW\n")
        .split_once("\n=== MATCHES")
        .map_or("", |(head, _)| head);
    let matches = after("=== MATCHES\n");
    (
        ocr,
        chars,
        preview.trim().to_owned(),
        matches.trim().to_owned(),
    )
}

/// What converting a file found: OCR used, characters, preview, and matching passages.
pub type Converted = (bool, usize, String, String);

/// One line naming a file: its name, type and size.
fn described(template: &str, file: &FileAttachment, bytes: usize) -> String {
    template
        .replace("{name}", &file.name)
        .replace("{kind}", kind(file))
        .replace("{size}", &size(bytes))
}

fn kind(file: &FileAttachment) -> &str {
    if file.media_type.is_empty() {
        "unknown type"
    } else {
        &file.media_type
    }
}

/// A byte count as a person reads it.
fn size(bytes: usize) -> String {
    #[allow(clippy::cast_precision_loss)]
    let kb = bytes as f64 / 1024.0;
    if kb < 1024.0 {
        format!("{kb:.0} KB")
    } else {
        format!("{:.1} MB", kb / 1024.0)
    }
}

impl Agent {
    /// Copies every attached file into the workspace and returns the prompt lines naming them.
    ///
    /// A file that cannot be read or written gets a line saying so, so the model never answers
    /// as if it had read it. Empty when nothing is attached.
    pub(super) async fn uploads(&self, ctx: &RequestContext, input: &str) -> Vec<String> {
        let mut lines = Vec::new();
        let session = workspace_session(ctx);
        for file in &ctx.files {
            match self.upload(ctx, &session, file, input).await {
                Ok((path, bytes, text)) => {
                    self.deps.trace.emit(
                        ctx,
                        TraceEvent::FileAttached {
                            name: file.name.clone(),
                            bytes,
                            path: Some(path.clone()),
                            error: None,
                        },
                    );
                    let template = match &text {
                        Some((_, chars, _, _)) if *chars > 0 => &self.prompt.upload_line,
                        _ => &self.prompt.upload_raw_line,
                    };
                    let (ocr, chars, preview, matches) = text.unwrap_or_default();
                    let matches = if matches.is_empty() {
                        "(no passage holds the words of the question)".to_owned()
                    } else {
                        matches
                    };
                    let read = if ocr { " (read by OCR)" } else { "" };
                    lines.push(
                        described(template, file, bytes)
                            .replace("{path}", &path)
                            .replace("{text_path}", &format!("{path}.txt"))
                            .replace("{chars}", &format!("{chars}{read}"))
                            .replace("{session}", &session)
                            .replace("{preview}", &preview)
                            .replace("{matches}", &matches),
                    );
                }
                Err(reason) => {
                    tracing::warn!(file = %file.name, %reason, "attached file not opened");
                    self.deps.trace.emit(
                        ctx,
                        TraceEvent::FileAttached {
                            name: file.name.clone(),
                            bytes: 0,
                            path: None,
                            error: Some(reason.clone()),
                        },
                    );
                    lines.push(
                        described(&self.prompt.upload_failed_line, file, 0)
                            .replace("{reason}", &reason),
                    );
                }
            }
        }
        lines
    }

    /// Downloads one file, writes it to the workspace, and converts it to text there.
    ///
    /// Returns its path, its size, and what the conversion found. A conversion that fails leaves
    /// the file in place with no text, for the model to read itself.
    async fn upload(
        &self,
        ctx: &RequestContext,
        session: &str,
        file: &FileAttachment,
        input: &str,
    ) -> Result<(String, usize, Option<Converted>), String> {
        let (Some(files), Some(sandbox)) = (&self.deps.files, &self.deps.sandbox) else {
            return Err("files cannot be opened here".to_owned());
        };
        if !sandbox.enabled() {
            return Err("files cannot be opened here".to_owned());
        }
        let bytes = files.fetch(ctx, file).await.map_err(|e| e.to_string())?;
        let name = format!("upload-{}", file.safe_name());
        let path = sandbox
            .put(ctx, session, &name, &bytes)
            .await
            .map_err(|e| e.to_string())?;
        let request = SandboxRequest {
            command: to_text(
                &path,
                &terms(input),
                self.cfg.upload_preview_chars,
                self.cfg.upload_match_chars,
                self.cfg.upload_ocr_pages,
            ),
            session: Some(session.to_owned()),
        };
        let text = match sandbox.run(ctx, &request).await {
            Ok(out) if out.exit_code == 0 => Some(converted(&out.stdout)),
            Ok(out) => {
                tracing::warn!(file = %file.name, stderr = %out.stderr, "attached file not converted");
                None
            }
            Err(error) => {
                tracing::warn!(file = %file.name, %error, "attached file not converted");
                None
            }
        };
        Ok((path, bytes.len(), text))
    }
}
