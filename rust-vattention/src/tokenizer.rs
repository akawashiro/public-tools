use crate::word;
use anyhow::{ensure, Context, Result};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

/// llama2.c's exported SentencePiece vocabulary and score-based BPE.
pub struct Tokenizer {
    pieces: Vec<Vec<u8>>,
    scores: Vec<f32>,
    ids: HashMap<Vec<u8>, usize>,
}

impl Tokenizer {
    pub fn load(path: impl AsRef<Path>, vocab_size: usize) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("open {}", path.as_ref().display()))?;
        Self::read(BufReader::new(file), vocab_size)
    }

    fn read(mut reader: impl Read, vocab_size: usize) -> Result<Self> {
        ensure!(
            (259..=10_000_000).contains(&vocab_size),
            "invalid vocabulary size"
        );
        let max_length = u32::from_le_bytes(word(&mut reader)?) as usize;
        ensure!(
            (1..=1_048_576).contains(&max_length),
            "invalid max token length"
        );
        let mut pieces = Vec::new();
        let mut scores = Vec::new();
        let mut ids = HashMap::new();
        for id in 0..vocab_size {
            let score = f32::from_le_bytes(word(&mut reader)?);
            ensure!(score.is_finite(), "non-finite tokenizer score");
            let len = u32::from_le_bytes(word(&mut reader)?) as usize;
            ensure!(len <= max_length, "token length exceeds tokenizer header");
            let mut piece = vec![0; len];
            reader.read_exact(&mut piece)?;
            ids.insert(piece.clone(), id);
            pieces.push(piece);
            scores.push(score);
        }
        ensure!(
            ids.contains_key(b" ".as_slice()),
            "missing dummy prefix token"
        );
        for byte in 0..=255u8 {
            ensure!(
                pieces[byte as usize + 3] == format!("<0x{byte:02X}>").as_bytes(),
                "incompatible byte fallback vocabulary"
            );
        }
        let mut extra = [0];
        ensure!(
            reader.read(&mut extra)? == 0,
            "tokenizer vocabulary size mismatch"
        );
        Ok(Self {
            pieces,
            scores,
            ids,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![1];
        if !text.is_empty() {
            tokens.push(self.ids[b" ".as_slice()]);
        }
        for c in text.chars() {
            let mut buffer = [0; 4];
            let bytes = c.encode_utf8(&mut buffer).as_bytes();
            if let Some(id) = self.ids.get(bytes) {
                tokens.push(*id);
            } else {
                tokens.extend(bytes.iter().map(|b| *b as usize + 3));
            }
        }
        loop {
            let mut best: Option<(usize, usize)> = None;
            for (i, pair) in tokens.windows(2).enumerate() {
                // BOS is structural and must never participate in a merge.
                if pair[0] <= 2 || pair[1] <= 2 {
                    continue;
                }
                let merged = [
                    self.pieces[pair[0]].as_slice(),
                    self.pieces[pair[1]].as_slice(),
                ]
                .concat();
                if let Some(&id) = self.ids.get(&merged) {
                    if best.is_none_or(|(_, old)| self.scores[id] > self.scores[old]) {
                        best = Some((i, id));
                    }
                }
            }
            let Some((i, id)) = best else {
                break;
            };
            tokens[i] = id;
            tokens.remove(i + 1);
        }
        tokens
    }

    /// Return bytes so UTF-8 byte fallback can be streamed without corruption.
    pub fn decode(&self, previous: usize, token: usize) -> Result<Vec<u8>> {
        let mut piece = self
            .pieces
            .get(token)
            .context("token outside tokenizer vocabulary")?
            .as_slice();
        if token <= 2 {
            return Ok(Vec::new());
        }
        if (3..259).contains(&token) {
            return Ok(vec![(token - 3) as u8]);
        }
        if previous == 1 && piece.starts_with(b" ") {
            piece = &piece[1..];
        }
        Ok(piece.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Tokenizer {
        let mut pieces = vec![b"<unk>".to_vec(), b"<s>".to_vec(), b"</s>".to_vec()];
        pieces.extend((0..=255u8).map(|b| format!("<0x{b:02X}>").into_bytes()));
        pieces.extend([
            b" ".to_vec(),
            b"a".to_vec(),
            b"b".to_vec(),
            b"ab".to_vec(),
            b" ab".to_vec(),
        ]);
        let mut data = 6u32.to_le_bytes().to_vec();
        for (i, p) in pieces.iter().enumerate() {
            data.extend((i as f32).to_le_bytes());
            data.extend((p.len() as u32).to_le_bytes());
            data.extend(p);
        }
        Tokenizer::read(data.as_slice(), pieces.len()).unwrap()
    }
    #[test]
    fn bpe_and_unicode_roundtrip() {
        let t = fixture();
        assert_eq!(t.encode("ab"), vec![1, 263]);
        for text in ["", "ab", "日本語 🦀", " a\nb"] {
            let tokens = t.encode(text);
            let bytes: Vec<u8> = tokens
                .windows(2)
                .flat_map(|p| t.decode(p[0], p[1]).unwrap())
                .collect();
            assert_eq!(String::from_utf8(bytes).unwrap(), text);
        }
    }
}
