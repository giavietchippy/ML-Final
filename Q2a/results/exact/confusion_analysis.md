# Exact confusion analysis (from saved test predictions)

## Test accuracy recomputed from predictions

| model | test acc (this run) | test acc (original) |
|---|---:|---:|
| baseline | 0.5998 | 0.5833 |
| tuned | 0.6844 | 0.6844 |

## Per-class test accuracy (recall)

| class | baseline | tuned | change |
|---|---:|---:|---:|
| airplane | 0.677 | 0.827 | +0.150 |
| automobile | 0.746 | 0.869 | +0.123 |
| bird | 0.424 | 0.455 | +0.031 |
| cat | 0.474 | 0.439 | -0.035 |
| deer | 0.399 | 0.586 | +0.187 |
| dog | 0.511 | 0.612 | +0.101 |
| frog | 0.650 | 0.714 | +0.064 |
| horse | 0.685 | 0.815 | +0.130 |
| ship | 0.721 | 0.717 | -0.004 |
| truck | 0.711 | 0.810 | +0.099 |

## baseline: most confused ordered pairs (true -> predicted)

| true | predicted | count |
|---|---|---:|
| dog | cat | 252 |
| cat | dog | 192 |
| bird | cat | 158 |
| frog | cat | 140 |
| deer | cat | 139 |

## baseline: most confused unordered pairs (both directions summed)

| class A | class B | count |
|---|---|---:|
| cat | dog | 444 |
| cat | frog | 230 |
| bird | cat | 228 |
| automobile | truck | 203 |
| airplane | ship | 186 |

## tuned: most confused ordered pairs (true -> predicted)

| true | predicted | count |
|---|---|---:|
| cat | dog | 174 |
| deer | horse | 168 |
| ship | airplane | 150 |
| bird | airplane | 131 |
| dog | cat | 118 |

## tuned: most confused unordered pairs (both directions summed)

| class A | class B | count |
|---|---|---:|
| cat | dog | 292 |
| automobile | truck | 206 |
| deer | horse | 200 |
| airplane | ship | 190 |
| bird | deer | 143 |
