"""The phase-8 corpus builder's two safety properties: Gutenberg boilerplate is stripped, and
any paragraph quoting Shakespeare (the validation set) is caught by the 8-gram guard."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "build_corpus", Path(__file__).resolve().parents[1] / "scripts" / "build_corpus.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

HAMLET = ("To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer "
          "The slings and arrows of outrageous fortune")


def banned_from(text):
    return bc.ngrams(bc.words(text))


def test_quotation_is_caught_regardless_of_case_and_punctuation():
    banned = banned_from(HAMLET)
    quote = 'As the Prince saith, "TO BE -- or not to be; that is the QUESTION," and so forth.'
    assert bc.ngrams(bc.words(quote)) & banned


def test_short_common_phrases_are_not_caught():
    banned = banned_from(HAMLET)
    para = "Whether it be nobler or not, the question is one of fortune and the mind."
    assert not (bc.ngrams(bc.words(para)) & banned)        # shares words, but no 8-gram


def test_strip_gutenberg_both_marker_styles():
    for start, end in [("*** START OF THE PROJECT GUTENBERG EBOOK FAUSTUS ***",
                        "*** END OF THE PROJECT GUTENBERG EBOOK FAUSTUS ***"),
                       ("***START OF THIS PROJECT GUTENBERG EBOOK VOLPONE***",
                        "***END OF THIS PROJECT GUTENBERG EBOOK VOLPONE***")]:
        raw = f"licence blah\r\n{start}\r\n\r\nBody line one.\r\nBody line two.\r\n{end}\r\nmore licence"
        assert bc.strip_gutenberg(raw) == "Body line one.\nBody line two.\n"
    assert bc.strip_gutenberg("no markers here") is None


def test_leak_mask_fills_short_speeches_between_hits_but_not_isolated_ones():
    banned = banned_from(HAMLET)
    hit = " ".join(bc.words(HAMLET))
    keys = ["fair cousin i am glad", hit, "away", "my lord", hit,       # 0..4: a leaked scene
            *["an innocent line of other verse"] * 12, hit, "go on"]     # 5..16 gap, 17 lone quote
    mask = bc.leak_mask(keys, banned, gap=10)
    assert mask[:5] == [False, True, True, True, True]   # speeches between two close hits dropped
    assert not any(mask[5:17])                          # a 13-paragraph gap is not filled
    assert mask[17] and not mask[18]                     # an isolated quotation flags only itself
