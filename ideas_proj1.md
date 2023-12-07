Ideas: Music generation

Simplest model, generation might not be awesome
Markov chain

- Current state is only depenedent on prev state
- State H, what is probability of going to A, B, C, D, E
- "transition matrix" matrix of probabilities from one to another at time
- Sample from prob. distribution
- instead of letters, state is notes/chords
  Goals:
- Code up markov chian for text
- Code order-N markov chain (ngrams but you are only generating one letter at a time, not chunks)
- MIDI datasets, or download songs and do that
- parse using midiutil or something
- Run markov chain, output to file
