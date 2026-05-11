import re
import emoji
from functools import lru_cache
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

_stemmer_cache = None
_stem_cache = {}


def get_stemmer():
    global _stemmer_cache
    if _stemmer_cache is None:
        factory = StemmerFactory()
        _stemmer_cache = factory.create_stemmer()
    return _stemmer_cache


@lru_cache(maxsize=50000)
def _stem_word(word):
    stemmer = get_stemmer()
    return stemmer.stem(word)


INDONESIAN_STOPWORDS = {
    'ada', 'adalah', 'adanya', 'akan', 'aku', 'anda', 'antara', 'apa', 'apabila',
    'atas', 'atau', 'bagai', 'bagaimana', 'bagi', 'bahkan', 'bahwa', 'banyak',
    'baru', 'bawah', 'beberapa', 'begitu', 'belum', 'berada', 'berakhiran',
    'berasal', 'berbagai', 'berdasrkan', 'berharap', 'beri', 'berikut', 'berkata',
    'bersama', 'bersamanya', 'besar', 'bila', 'bisa', 'boleh', 'buat', 'bukan',
    'buruk', 'cara', 'dalam', 'dan', 'dapat', 'dari', 'daripada', 'dekat',
    'demi', 'dengan', 'demikian', 'di', 'dia', 'diantara', 'diberikan', 'dibuat',
    'digunakan', 'dilakukan', 'dimana', 'diri', 'disamping', 'disebut', 'disini',
    'distribusi', 'dll', 'dr', 'engkau', 'gimana', 'hal', 'hampir', 'hanya',
    'harus', 'hendak', 'hingga', 'ia', 'ialah', 'ini', 'inilah', 'itu', 'itulah',
    'jadi', 'jangan', 'jarang', 'jauh', 'jenis', 'jika', 'juga', 'kah', 'kalau',
    'kami', 'kamu', 'karena', 'kata', 'katanya', 'ke', 'kecil', 'kedua', 'keluar',
    'kemudian', 'kenapa', 'kepada', 'ketika', 'kira', 'kita', 'kok', 'lagi',
    'lain', 'lanjut', 'lama', 'lalu', 'lebih', 'lewat', 'luar', 'macam', 'maka',
    'mampu', 'mana', 'masa', 'masalah', 'masih', 'maupun', 'melalui', 'melihat',
    'memang', 'membuat', 'memiliki', 'menjadi', 'mengapa', 'mengatakan', 'merupakan',
    'meski', 'meskipun', 'mereka', 'mungkin', 'nah', 'namun', 'oleh', 'orang',
    'pada', 'paling', 'para', 'pergi', 'perlu', 'pertama', 'pula', 'pun', 'saat',
    'saja', 'salah', 'saling', 'sama', 'sampai', 'sangat', 'satu', 'saya', 'se',
    'sebagai', 'sebelum', 'sebuah', 'secara', 'sedang', 'sedikit', 'segala',
    'sehingga', 'sejak', 'sekali', 'sekarang', 'selalu', 'selama', 'seluruh',
    'semakin', 'sembari', 'semua', 'sendiri', 'seorang', 'sepanjang', 'seperti',
    'sering', 'sesuatu', 'sesudah', 'setelah', 'setiap', 'sini', 'situ', 'suatu',
    'sudah', 'supaya', 'tadi', 'tampa', 'tanpa', 'tapi', 'telah', 'tentang',
    'terdapat', 'terhadap', 'terlalu', 'tersebut', 'terus', 'tetaplah', 'tidak',
    'tu', 'turun', 'untuk', 'waktu', 'ya', 'yaitu', 'yang', 'a', 'an', 'the',
    'in', 'is', 'it', 'and', 'of', 'to', 'be', 'are',
}

URL_PATTERN = re.compile(r'\[URL\]|https?://\S+|www\.\S+', re.IGNORECASE)
USERNAME_PATTERN = re.compile(r'\[USERNAME\]|@\w+', re.IGNORECASE)
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#\w+')


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub('', text)


def remove_usernames(text: str) -> str:
    return USERNAME_PATTERN.sub('', text)


def extract_emojis(text: str) -> str:
    chars = list(text)
    emoji_chars = ''.join(c for c in chars if c in emoji.EMOJI_DATA or c == '️')
    return emoji_chars


def normalize_emojis(emoji_text: str) -> str:
    return emoji.demojize(emoji_text, delimiters=('', ''))


def normalize_slang(text: str) -> str:
    slang_map = {
        'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak', 'tdk': 'tidak',
        'nggak': 'tidak', 'enggak': 'tidak', 'tak': 'tidak',
        'kagak': 'tidak', 'ndak': 'tidak', 'g': 'tidak',
        'dgn': 'dengan', 'dg': 'dengan',
        'yg': 'yang', 'yng': 'yang',
        'jg': 'juga', 'jga': 'juga',
        'bs': 'bisa', 'bsa': 'bisa',
        'trs': 'terus',
        'udh': 'sudah', 'udah': 'sudah',
        'blm': 'belum',
        'tp': 'tapi', 'tpi': 'tapi',
        'krn': 'karena',
        'knp': 'kenapa', 'knpa': 'kenapa',
        'sy': 'saya', 'aq': 'aku', 'gw': 'saya', 'gue': 'saya',
        'km': 'kamu', 'lo': 'kamu', 'lu': 'kamu', 'loe': 'kamu',
        'bg': 'bang', 'bgt': 'banget',
        'aja': 'saja',
        'bgt': 'sangat',
        'mksd': 'maksud', 'mksud': 'maksud',
        'dri': 'dari',
        'dl': 'dalam',
        'dlu': 'dulu',
        'skrg': 'sekarang', 'skrng': 'sekarang',
        'tdr': 'tidur',
        'pd': 'pada',
        'dpt': 'dapat',
        'jgn': 'jangan',
        'klo': 'kalau',
        'mlh': 'malah',
        'utk': 'untuk',
        'org': 'orang',
        'trnyata': 'ternyata',
        'pdhl': 'padahal',
        'bkn': 'bukan',
        'drpd': 'daripada',
        'sm': 'sama',
        'kpn': 'kapan',
        'hrs': 'harus',
        'bgs': 'bagus',
        'krg': 'kurang',
        'bnyk': 'banyak',
        'sdh': 'sudah',
        'tdk': 'tidak',
        'jg': 'juga',
        'sy': 'saya',
        'km': 'kamu',
        'bh': 'bahwa',
        'jln': 'jalan',
        'spt': 'seperti',
        'mks': 'maksud',
        'dg': 'dengan',
        'dr': 'dari',
        'dl': 'dalam',
        'br': 'baru',
        'bsk': 'besok',
        'rmh': 'rumah',
        'msk': 'masuk',
        'klr': 'keluar',
        'pn': 'pun',
        'dg': 'dengan',
        'dl': 'dalam',
        'jg': 'juga',
        'pd': 'pada',
        'utk': 'untuk',
        'tdk': 'tidak',
        'sy': 'saya',
        'km': 'kamu',
        'sya': 'saya',
        'aq': 'aku',
        'gw': 'saya',
        'lo': 'kamu',
        'lu': 'kamu',
        'thx': 'terima kasih',
        'thks': 'terima kasih',
        'makasih': 'terima kasih',
        'mksh': 'terima kasih',
        'trims': 'terima kasih',
        'ty': 'terima kasih',
        'tq': 'terima kasih',
        'ngentod': '',
        'anjing': '',
        'bangsat': '',
        'bgst': '',
        'kontol': '',
        'memek': '',
        'bego': 'bodoh',
        'tolol': 'bodoh',
        'goblok': 'bodoh',
        'geblek': 'bodoh',
        'dongo': 'bodoh',
        'bejad': 'buruk',
        'brengsek': 'buruk',
        'kampret': 'buruk',
        'jancok': '',
        'jancuk': '',
        'asu': '',
        'babi': '',
        'bacot': '',
        'koplak': 'bodoh',
        'koplok': 'bodoh',
        'sarap': '',
        'pepek': '',
        'ngaco': 'salah',
        'ngehe': '',
        'cok': '',
        'cuk': '',
        'ngtd': '',
        'ngt': '',
        'smpe': 'sampai',
        'smg': 'semoga',
    }
    words = text.split()
    normalized = [slang_map.get(w.lower(), w) for w in words]
    return ' '.join(w for w in normalized if w)


def preprocess_text(text: str, remove_url: bool = True, remove_username: bool = True,
                    extract_emoji: bool = True, normalize_emoji: bool = True,
                    normalize_slang_text: bool = True, stem: bool = True,
                    remove_stopwords: bool = True) -> str:
    if remove_url:
        text = remove_urls(text)
    if remove_username:
        text = remove_usernames(text)
    if extract_emoji or normalize_emoji:
        text_emojis = extract_emojis(text)
        if normalize_emoji:
            text_emojis = normalize_emojis(text_emojis)
        if '️' in text and text_emojis:
            pass
    if normalize_slang_text:
        text = normalize_slang(text)
    if stem:
        words = text.split()
        text = ' '.join(_stem_word(w) for w in words)
    if remove_stopwords:
        words = text.split()
        text = ' '.join(w for w in words if w.lower() not in INDONESIAN_STOPWORDS)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_for_traditional(text: str) -> str:
    return preprocess_text(
        text,
        remove_url=True, remove_username=True,
        extract_emoji=True, normalize_emoji=True,
        normalize_slang_text=True, stem=True,
        remove_stopwords=True,
    )


def preprocess_for_deep_learning(text: str) -> str:
    return preprocess_text(
        text,
        remove_url=False, remove_username=False,
        extract_emoji=True, normalize_emoji=True,
        normalize_slang_text=True, stem=False,
        remove_stopwords=False,
    )


def preprocess_for_transformers(text: str) -> str:
    return preprocess_text(
        text,
        remove_url=True, remove_username=True,
        extract_emoji=False, normalize_emoji=False,
        normalize_slang_text=True, stem=False,
        remove_stopwords=False,
    )


def preprocess_batch(texts, mode='traditional', n_jobs=-1):
    """
    Parallel batch preprocessing using multiprocessing.
    mode: 'traditional', 'deep_learning', or 'transformers'
    n_jobs: number of workers (default: all cores)
    """
    from concurrent.futures import ProcessPoolExecutor

    fn = {
        'traditional': preprocess_for_traditional,
        'deep_learning': preprocess_for_deep_learning,
        'transformers': preprocess_for_transformers,
    }[mode]

    with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        results = list(executor.map(fn, texts))
    return results