import re
import emoji
import saka
from saka.dict.stopwords import get_stopwords as _saka_get_stopwords
import emot as _emot_lib

_STOPWORDS = _saka_get_stopwords('id')

_emot_detector = None


def _get_emot_detector():
    global _emot_detector
    if _emot_detector is None:
        _emot_detector = _emot_lib.core.emot()
    return _emot_detector


URL_PATTERN = re.compile(r'\[URL\]|https?://\S+|www\.\S+|t\.co/\S+', re.IGNORECASE)
USERNAME_PATTERN = re.compile(r'\[USERNAME\]|@\w+', re.IGNORECASE)
HASHTAG_PATTERN = re.compile(r'#\w+')


def _convert_emoji(text: str) -> str:
    text = emoji.demojize(text, language='id')
    text = text.replace('_', ' ')
    return text


def _convert_emoticon(text: str) -> str:
    try:
        detector = _get_emot_detector()
        result = detector.emoticons(text)
        if result and result.get('flag') and 'value' in result and 'mean' in result:
            for val, mean in zip(result['value'], result['mean']):
                text = text.replace(val, f' {mean.lower()} ')
    except Exception:
        pass
    return text


def preprocess_text(
    text: str,
    remove_url: bool = True,
    remove_username: bool = True,
    remove_hashtag: bool = True,
    convert_emoji: bool = True,
    convert_emoticon: bool = True,
    normalize_slang: bool = True,
    remove_stopword: bool = True,
    morphological_analysis: bool = True,
    lowercase: bool = True,
    remove_extra_spaces: bool = True,
) -> str:
    if not text or not isinstance(text, str):
        return ''

    if lowercase:
        text = text.lower()

    if remove_url:
        text = URL_PATTERN.sub('', text)

    if remove_username:
        text = USERNAME_PATTERN.sub('', text)

    if remove_hashtag:
        text = HASHTAG_PATTERN.sub('', text)

    if convert_emoji:
        text = _convert_emoji(text)

    if convert_emoticon:
        text = _convert_emoticon(text)

    if normalize_slang:
        text = saka.normalize(text)

    tokens = saka.tokenize(text)

    clean_tokens = []
    for tok in tokens:
        if remove_stopword and tok.lower() in _STOPWORDS:
            continue
        if not any(c.isalpha() for c in tok):
            continue
        if morphological_analysis and tok.isalpha():
            analysis = saka.analyze(tok)
            root = analysis.get('root', tok)
            clean_tokens.append(root)
        else:
            clean_tokens.append(tok)

    cleaned = ' '.join(clean_tokens)
    if remove_extra_spaces:
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


def preprocess_for_traditional(text: str) -> str:
    return preprocess_text(
        text,
        remove_url=True,
        remove_username=True,
        remove_hashtag=True,
        convert_emoji=True,
        convert_emoticon=True,
        normalize_slang=True,
        remove_stopword=True,
        morphological_analysis=True,
    )


def preprocess_for_deep_learning(text: str) -> str:
    return preprocess_text(
        text,
        remove_url=False,
        remove_username=False,
        remove_hashtag=False,
        convert_emoji=True,
        convert_emoticon=True,
        normalize_slang=True,
        remove_stopword=False,
        morphological_analysis=False,
    )


def preprocess_for_transformers(text: str) -> str:
    return preprocess_text(
        text,
        remove_url=True,
        remove_username=True,
        remove_hashtag=True,
        convert_emoji=False,
        convert_emoticon=False,
        normalize_slang=True,
        remove_stopword=False,
        morphological_analysis=False,
    )


def preprocess_batch(texts, mode='traditional', n_jobs=-1):
    fn = {
        'traditional': preprocess_for_traditional,
        'deep_learning': preprocess_for_deep_learning,
        'transformers': preprocess_for_transformers,
    }[mode]
    return [fn(t) for t in texts]
