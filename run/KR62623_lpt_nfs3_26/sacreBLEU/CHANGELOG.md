# VERSION HISTORY

- 1.4.2 (2019-10-11)
  - Tokenization variant omitted from the chrF signature; it is relevant only for BLEU (thanks to Martin Popel)
  - Bugfix: call to sentence_bleu (thanks to Rachel Bawden)
  - Documentation example for Python API (thanks to Vlad Lyalin)
  - Calls to corpus_chrf and sentence_chrf now return a an object instead of a float (use result.score)

- 1.4.1 (2019-09-11)
   - Added sentence-level scoring via -sl (--sentence-level)

- 1.4.0 (2019-09-10)
   - Many thanks to Martin Popel for all the changes below!
   - Added evaluation on concatenated test sets (e.g., `-t wmt17,wmt18`).
     Works as long as they all have the same language pair.
   - Added `sacrebleu --origlang` (both for evaluation on a subset and for `--echo`).
     Note that while echoing prints just the subset, evaluation expects the complete
     test set (and just skips the irrelevant parts).
   - Added `sacrebleu --detail` for breakdown by domain-specific subsets of the test sets.
     (Available for WMT19).
   - Minor changes
     - Improved display of `sacrebleu -h`
     - Added `sacrebleu --list`
     - Code refactoring
     - Documentation and tests updates
     - Fixed a race condition bug (`os.makedirs(outdir, exist_ok=True)` instead of `if os.path.exists`)

- 1.3.7 (2019-07-12)
   - Lazy loading of regexes cuts import time from ~1s to nearly nothing (thanks, @louismartin!)
   - Added a simple (non-atomic) lock on downloading
   - Can now read multiple refs from a single tab-delimited file.
     You need to pass `--num-refs N` to tell it to run the split.
     Only works with a single reference file passed from the command line.

- 1.3.6 (2019-06-10)
   - Removed another f-string for Python 3.5 compatibility

- 1.3.5 (2019-06-07)
   - Restored Python 3.5 compatibility

- 1.3.4 (2019-05-28)
   - Added MTNT 2019 test sets
   - Added a BLEU object

- 1.3.3 (2019-05-08)
   - Added WMT'19 test sets

- 1.3.2 (2018-04-24)
   - Bugfix in test case (thanks to Adam Roberts, @adarob)
   - Passing smoothing method through `sentence_bleu`

- 1.3.1 (2019-03-20)
   - Added another smoothing approach (add-k) and a command-line option for choosing the smoothing method
     (`--smooth exp|floor|add-n|none`) and the associated value (`--smooth-value`), when relevant.
   - Changed interface to some functions (backwards incompatible)
     - 'smooth' is now 'smooth_method'
     - 'smooth_floor' is now 'smooth_value'

- 1.2.21 (19 March 2019)
   - Ctrl-M characters are now treated as normal characters, previously treated as newline.

- 1.2.20 (28 February 2018)
   - Tokenization now defaults to "zh" when language pair is known

- 1.2.19 (19 February 2019)
   - Updated checksum for wmt19/dev (seems to have changed)

- 1.2.18 (19 February 2019)
   - Fixed checksum for wmt17/dev (copy-paste error)

- 1.2.17 (6 February 2019)
   - Added kk-en and en-kk to wmt19/dev

- 1.2.16 (4 February 2019)
   - Added gu-en and en-gu to wmt19/dev

- 1.2.15 (30 January 2019)
   - Added MD5 checksumming of downloaded files for all datasets.

- 1.2.14 (22 January 2019)
   - Added mtnt1.1/train mtnt1.1/valid mtnt1.1/test data from [MTNT](http://www.cs.cmu.edu/~pmichel1/mtnt/)

- 1.2.13 (22 January 2019)
   - Added 'wmt19/dev' task for 'lt-en' and 'en-lt' (development data for new tasks).
   - Added MD5 checksum for downloaded tarballs.

- 1.2.12 (8 November 2018)
   - Now outputs only only digit after the decimal

- 1.2.11 (29 August 2018)
   - Added a function for sentence-level, smoothed BLEU

- 1.2.10 (23 May 2018)
   - Added wmt18 test set (with references)

- 1.2.9 (15 May 2018)
   - Added zh-en, en-zh, tr-en, and en-tr datasets for wmt18/test-ts

- 1.2.8 (14 May 2018)
   - Added wmt18/test-ts, the test sources (only) for [WMT18](http://statmt.org/wmt18/translation-task.html)
   - Moved README out of `sacrebleu.py` and the CHANGELOG into a separate file

- 1.2.7 (10 April 2018)
   - fixed another locale issue (with --echo)
   - grudgingly enabled `-tok none` from the command line

- 1.2.6 (22 March 2018)
   - added wmt17/ms (Microsoft's [additional ZH-EN references](https://github.com/MicrosoftTranslator/Translator-HumanParityData)).
     Try `sacrebleu -t wmt17/ms --cite`.
   - `--echo ref` now pastes together all references, if there is more than one

- 1.2.5 (13 March 2018)
   - added wmt18/dev datasets (en-et and et-en)
   - fixed logic with --force
   - locale-independent installation
   - added "--echo both" (tab-delimited)

- 1.2.3 (28 January 2018)
   - metrics (`-m`) are now printed in the order requested
   - chrF now prints a version string (including the beta parameter, importantly)
   - attempt to remove dependence on locale setting

- 1.2 (17 January 2018)
   - added the chrF metric (`-m chrf` or `-m bleu chrf` for both)
     See 'CHRF: character n-gram F-score for automatic MT evaluation' by Maja Popovic (WMT 2015)
     [http://www.statmt.org/wmt15/pdf/WMT49.pdf]
   - added IWSLT 2017 test and tuning sets for DE, FR, and ZH
     (Thanks to Mauro Cettolo and Marcello Federico).
   - added `--cite` to produce the citation for easy inclusion in papers
   - added `--input` (`-i`) to set input to a file instead of STDIN
   - removed accent mark after objection from UN official

- 1.1.7 (27 November 2017)
   - corpus_bleu() now raises an exception if input streams are different lengths
   - thanks to Martin Popel for:
      - small bugfix in tokenization_13a (not affecting WMT references)
      - adding `--tok intl` (international tokenization)
   - added wmt17/dev and wmt17/dev sets (for languages intro'd those years)

- 1.1.6 (15 November 2017)
   - bugfix for tokenization warning

- 1.1.5 (12 November 2017)
   - added -b option (only output the BLEU score)
   - removed fi-en from list of WMT16/17 systems with more than one reference
   - added WMT16/tworefs and WMT17/tworefs for scoring with both en-fi references

- 1.1.4 (10 November 2017)
   - added effective order for sentence-level BLEU computation
   - added unit tests from sockeye

- 1.1.3 (8 November 2017).
   - Factored code a bit to facilitate API:
      - compute_bleu: works from raw stats
      - corpus_bleu for use from the command line
      - raw_corpus_bleu: turns off tokenization, command-line sanity checks, floor smoothing
   - Smoothing (type 'exp', now the default) fixed to produce mteval-v13a.pl results
   - Added 'floor' smoothing (adds 0.01 to 0 counts, more versatile via API), 'none' smoothing (via API)
   - Small bugfixes, windows compatibility (H/T Christian Federmann)

- 1.0.3 (4 November 2017).
   - Contributions from Christian Federmann:
      - Added explicit support for encoding
      - Fixed Windows support
      - Bugfix in handling reference length with multiple refs

- version 1.0.1 (1 November 2017).
   - Small bugfix affecting some versions of Python.
   - Code reformatting due to Ozan Çağlayan.

- version 1.0 (23 October 2017).
   - Support for WMT 2008--2017.
   - Single tokenization (v13a) with lowercase fix (proper lower() instead of just A-Z).
   - Chinese tokenization.
   - Tested to match all WMT17 scores on all arcs.
