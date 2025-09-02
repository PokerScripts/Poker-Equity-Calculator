#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poker-equity-calculator.py
CLI-калькулятор equity (шансов на банк) для Техасского Холдема.

Особенности:
- Монте-Карло по умолчанию, с seed, прогресс-баром и таймаутом.
- До 6 игроков (герой + 1–5 оппонентов).
- Частично заданная доска (0–5 карт).
- Руки оппонентов: фиксированные или диапазоны (PokerStove-подобный синтаксис).
- Вывод: человеко-читаемый или JSON.
- Точный перебор (exact) поддержан для случая, когда ВСЕ карманные руки фиксированы (по одной комбо на игрока).
  В остальных случаях скрипт вежливо откажет от exact и предложит Monte-Carlo.

Зависимости: только стандартная библиотека Python 3.9+

Примеры:
    # Герой: AsKs, оппонент: QhQd, доска пустая, 200k симуляций
    python poker-equity-calculator.py --hero AsKs --villain QhQd --iters 200000

    # Герой: AhKh, оппонент диапазон: "JJ+,AK", флоп: "QsJh2h", 500k итераций
    python poker-equity-calculator.py --hero AhKh --villain "JJ+,AK" --board QsJh2h --iters 500000

    # Мультипот: герой AsKd, два оппа: "random" и "22-99,ATs+,KQs", 300k итераций, JSON-вывод
    python poker-equity-calculator.py --hero AsKd --villains "random" "22-99,ATs+,KQs" --iters 300000 --json

    # Точный перебор для heads-up при пустой доске (если доступно)
    python poker-equity-calculator.py --hero 7h7d --villain AdKd --mode exact

    # Фиксация seed и прогресс-бар
    python poker-equity-calculator.py --hero AhAd --villain random --iters 1000000 --seed 42 --progress
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence, Set, Dict

# ------------------------------------------------------------
# Константы карт и утилиты
# ------------------------------------------------------------

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_TO_IDX = {r: i for i, r in enumerate(RANKS)}
IDX_TO_RANK = {i: r for i, r in enumerate(RANKS)}
SUIT_TO_IDX = {s: i for i, s in enumerate(SUITS)}
IDX_TO_SUIT = {i: s for i, s in enumerate(SUITS)}

# Категории комбинаций (чем больше, тем сильнее)
# 8: straight flush, 7: quads, 6: full house, 5: flush,
# 4: straight, 3: trips, 2: two pair, 1: one pair, 0: high card
CAT_NAMES = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"
]


def card_to_int(card: str) -> int:
    """Преобразование строки карты 'As' в int 0..51 (ранг + 13*масть)."""
    if len(card) != 2 or card[0] not in RANK_TO_IDX or card[1] not in SUIT_TO_IDX:
        raise ValueError(f"Неверная карта: {card}")
    r = RANK_TO_IDX[card[0]]
    s = SUIT_TO_IDX[card[1]]
    return r + 13 * s


def int_to_card(ci: int) -> str:
    """Обратное преобразование int -> 'As'."""
    r = ci % 13
    s = ci // 13
    return f"{IDX_TO_RANK[r]}{IDX_TO_SUIT[s]}"


def parse_cards_concat(s: str) -> List[int]:
    """Парсинг склеенных карт 'QsJh2h' -> [Q♠, J♥, 2♥]."""
    if not s:
        return []
    if len(s) % 2 != 0:
        raise ValueError(f"Неверная длина строки карт: '{s}'")
    out = []
    for i in range(0, len(s), 2):
        out.append(card_to_int(s[i:i+2]))
    return out


def deck52() -> List[int]:
    """Полная колода 52 карты."""
    return list(range(52))


def remove_cards(deck: List[int], cards: Sequence[int]) -> None:
    """Удалить карты из колоды (in-place)."""
    s = set(cards)
    k = 0
    for c in deck:
        if c not in s:
            deck[k] = c
            k += 1
    del deck[k:]


def comb(n: int, k: int) -> int:
    """Число сочетаний C(n, k)."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


# ------------------------------------------------------------
# Работа с диапазонами
# ------------------------------------------------------------

@dataclass
class RangeSpec:
    """Описывает диапазон карманных карт игрока."""
    label: str                  # строка диапазона, как указал пользователь (или 'random' / конкретная рука)
    is_random: bool             # случайная рука из оставшейся колоды
    combos: Optional[List[Tuple[int, int]]]  # список конкретных 2-карточных комбо (c1 < c2) или None, если random


HAND2_RE = re.compile(r"^([2-9TJQKA][cdhs]){2}$")


def try_parse_exact_hand(s: str) -> Optional[Tuple[int, int]]:
    """Если строка — ровно две карты, вернуть (c1,c2), иначе None."""
    if not s:
        return None
    s = s.strip()
    if HAND2_RE.match(s):
        c1 = card_to_int(s[:2])
        c2 = card_to_int(s[2:])
        if c1 == c2:
            raise ValueError(f"Дубликат карты в руке: {s}")
        if c2 < c1:
            c1, c2 = c2, c1
        return (c1, c2)
    return None


def all_combos_for_ranks(r1: int, r2: int, suitedness: str) -> List[Tuple[int, int]]:
    """
    Возвращает все конкретные комбо для рангов (r1, r2).
    suitedness: 's' (одномастные), 'o' (разномастные), 'x' (любой).
    Для пары r1==r2 suitedness игнорируется.
    """
    combos = []
    if r1 == r2:
        # Пара: выбрать 2 масти из 4 -> 6 комбо
        suits = range(4)
        for s1 in suits:
            for s2 in suits:
                if s2 <= s1:
                    continue
                c1 = r1 + 13 * s1
                c2 = r2 + 13 * s2
                combos.append((min(c1, c2), max(c1, c2)))
        return combos

    # Не пара
    for s1 in range(4):
        for s2 in range(4):
            if r1 != r2:
                same_suit = (s1 == s2)
                if suitedness == 's' and not same_suit:
                    continue
                if suitedness == 'o' and same_suit:
                    continue
            c1 = r1 + 13 * s1
            c2 = r2 + 13 * s2
            if c1 == c2:
                continue
            if c2 < c1:
                c1, c2 = c2, c1
            combos.append((c1, c2))
    # Удалим дубликаты (могут возникать при r1!=r2, но на всякий случай)
    combos = list(set(combos))
    combos.sort()
    return combos


def expand_token_to_combos(tok: str) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Разбор одного токена диапазона в список комбо.
    Возвращает (is_random, combos). Если is_random=True, combos игнорируется.

    Поддержано:
        - 'random'
        - точная рука: 'AsKs'
        - пары: 'JJ', 'JJ+', '22-99'
        - непары: 'AK', 'AKs', 'AKo', 'ATs+', 'A2s-A5s'
        - '+' для непар: первый ранг фикс, второй растёт до (первый ранг - 1).
          Примеры: 'ATs+' -> ATs,AJs,AQs,AKs; 'KJo+' -> KJo,KQo
        - Для диапазона через '-', требуем одинаковую мастьность (s/o/ничего) и одинаковый первый ранг.
    """
    tok = tok.strip()
    if not tok:
        return (False, [])
    if tok.lower() == "random":
        return (True, [])

    # Точная рука
    hand = try_parse_exact_hand(tok)
    if hand:
        return (False, [hand])

    # Пары: 'JJ' или 'JJ+' или '22-99'
    m = re.fullmatch(r"([2-9TJQKA])\1(\+)?", tok)
    if m:
        r = RANK_TO_IDX[m.group(1)]
        ranks = list(range(r, 13 + 1))  # до A (индекс 12)
        # ограничим до 12 (A)
        ranks = [x for x in ranks if x <= 12]
        if not m.group(2):  # без '+'
            ranks = [r]
        combos = []
        for rr in ranks:
            combos.extend(all_combos_for_ranks(rr, rr, 'x'))
        return (False, combos)

    m = re.fullmatch(r"([2-9TJQKA])\1-([2-9TJQKA])\2", tok)
    if m:
        r1 = RANK_TO_IDX[m.group(1)]
        r2 = RANK_TO_IDX[m.group(2)]
        if r1 > r2:
            lo, hi = r1, r2
        else:
            lo, hi = r2, r1
        # пары возрастают по силе (22-99): от нижней к верхней
        lo, hi = min(r1, r2), max(r1, r2)
        combos = []
        for rr in range(lo, hi + 1):
            combos.extend(all_combos_for_ranks(rr, rr, 'x'))
        return (False, combos)

    # Непары: базовый вид 'AK', 'AKs', 'AKo' с опциональным '+'
    m = re.fullmatch(r"([2-9TJQKA])([2-9TJQKA])(s|o)?(\+)?", tok)
    if m:
        a = RANK_TO_IDX[m.group(1)]
        b = RANK_TO_IDX[m.group(2)]
        if a == b:
            raise ValueError(f"Токен '{tok}' похож на пару — используйте вид 'JJ'/'JJ+'/'22-99'")
        if a < b:
            # Нормируем в вид XY где X > Y по силе (A>K>Q>...)
            a, b = b, a
        so = m.group(3) or 'x'
        plus = bool(m.group(4))
        combos = []
        # '+' означает увеличение нижнего ранга от b до (a-1) включительно
        b_range = range(b, a) if plus else [b]
        for bb in b_range:
            combos.extend(all_combos_for_ranks(a, bb, so))
        return (False, combos)

    # Диапазон по непарам через '-': требуем одинаковую мастьность и одинаковый первый ранг
    m = re.fullmatch(r"([2-9TJQKA])([2-9TJQKA])(s|o)?-([2-9TJQKA])([2-9TJQKA])\3", tok)
    if m:
        a1 = RANK_TO_IDX[m.group(1)]
        b1 = RANK_TO_IDX[m.group(2)]
        a2 = RANK_TO_IDX[m.group(4)]
        b2 = RANK_TO_IDX[m.group(5)]
        so = m.group(3) or 'x'
        # Нормируем пары так, чтобы первый ранг был больше второго
        if a1 < b1:
            a1, b1 = b1, a1
        if a2 < b2:
            a2, b2 = b2, a2
        if a1 != a2:
            raise ValueError(f"Для диапазона через '-' ожидается одинаковый старший ранг: '{tok}'")
        lo, hi = min(b1, b2), max(b1, b2)
        combos = []
        for bb in range(lo, hi + 1):
            if bb >= a1:
                continue  # нельзя чтобы младший ранг >= старшего
            combos.extend(all_combos_for_ranks(a1, bb, so))
        return (False, combos)

    raise ValueError(f"Не смог разобрать токен диапазона: '{tok}'")


def parse_range(expr: str) -> RangeSpec:
    """Парсит выражение диапазона в RangeSpec."""
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Пустой диапазон.")
    if expr.lower() == "random":
        return RangeSpec(label="random", is_random=True, combos=None)

    # Разбиваем по запятым/пробелам
    raw_tokens = re.split(r"[,\s]+", expr.strip())
    is_random_any = False
    combo_set: Set[Tuple[int, int]] = set()
    for tok in raw_tokens:
        if not tok:
            continue
        is_rand, combos = expand_token_to_combos(tok)
        if is_rand:
            is_random_any = True
        else:
            combo_set.update(combos)
    if is_random_any and combo_set:
        # Микс random + конкретные комбо — неоднозначно.
        # Для простоты запретим такую смесь (можно снять ограничение при желании).
        raise ValueError("Нельзя смешивать 'random' с конкретными руками/диапазонами.")
    if is_random_any:
        return RangeSpec(label="random", is_random=True, combos=None)
    combos_sorted = list(combo_set)
    combos_sorted.sort()
    if not combos_sorted:
        raise ValueError(f"Пустой диапазон после разбора: '{expr}'")
    return RangeSpec(label=expr, is_random=False, combos=combos_sorted)


# ------------------------------------------------------------
# Оценка силы руки (7 карт -> ранговый кортеж)
# ------------------------------------------------------------

def hand_rank_7(cards7: Sequence[int]) -> Tuple[int, Tuple[int, ...]]:
    """
    Оценка лучшей 5-карточной руки из 7 карт.
    Возвращает (категория, ключ), где ключ — кортеж, достаточный для сравнения рук.
    Чем больше кортеж (лексикографически), тем сильнее рука.
    """
    # Подсчёты по рангам и мастям
    rank_counts = [0] * 13
    suit_counts = [0] * 4
    # Для детектирования флеша и стрит-флеша — маски рангов по каждой масти
    suit_rank_mask = [0] * 4
    rank_mask = 0

    ranks_by_suit = {0: [], 1: [], 2: [], 3: []}

    for c in cards7:
        r = c % 13
        s = c // 13
        rank_counts[r] += 1
        suit_counts[s] += 1
        suit_rank_mask[s] |= (1 << r)
        rank_mask |= (1 << r)
        ranks_by_suit[s].append(r)

    # Проверка флеша (5+ карт одной масти)
    flush_suit = -1
    for s in range(4):
        if suit_counts[s] >= 5:
            flush_suit = s
            break

    # Детекция стрита (возвращает старшую карту стрита или -1)
    def highest_straight_from_mask(mask: int) -> int:
        # Учтём wheel: A-2-3-4-5 (A как 12)
        wheel_mask = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
        if (mask & wheel_mask) == wheel_mask:
            hi = 3  # пятёрка (индекс 3) — старшая в A-5
        else:
            hi = -1
            # Ищем 5 подряд: от A (12) до 4 (3) — проверяем хвосты
            # Удобнее проверять от 12 вниз к 4: окно из 5 бит
            for top in range(12, 3, -1):  # 12..4
                window = 0
                for k in range(5):
                    window |= (1 << (top - k))
                if (mask & window) == window:
                    hi = top
                    break
        return hi

    # Сначала проверим стрит-флеш
    if flush_suit != -1:
        sf_hi = highest_straight_from_mask(suit_rank_mask[flush_suit])
        if sf_hi != -1:
            return (8, (sf_hi,))  # Straight Flush старшая карта

    # Каре?
    quad_rank = -1
    for r in range(12, -1, -1):
        if rank_counts[r] == 4:
            quad_rank = r
            break
    if quad_rank != -1:
        # Киккер — максимальный из оставшихся
        kick = max([rr for rr in range(12, -1, -1) if rr != quad_rank and rank_counts[rr] > 0])
        return (7, (quad_rank, kick))

    # Фулл-хаус?
    trips = -1
    pairs = []
    for r in range(12, -1, -1):
        if rank_counts[r] >= 3 and trips == -1:
            trips = r
        elif rank_counts[r] >= 2:
            pairs.append(r)
    if trips != -1 and (pairs or any(rank_counts[r] >= 2 and r != trips for r in range(13))):
        # Найдём лучшую пару для фулл-хауса
        best_pair = -1
        for r in range(12, -1, -1):
            if r == trips:
                if rank_counts[r] >= 2 and any(rank_counts[x] >= 2 for x in range(13) if x != r):
                    # когда две тройки, одну используем как пару
                    continue
            if r != trips and rank_counts[r] >= 2:
                best_pair = r
                break
        if best_pair == -1:
            # второй трипс -> используем как пару
            for r in range(12, -1, -1):
                if r != trips and rank_counts[r] >= 3:
                    best_pair = r
                    break
        if best_pair != -1:
            return (6, (trips, best_pair))

    # Флеш?
    if flush_suit != -1:
        rs = sorted(ranks_by_suit[flush_suit], reverse=True)
        top5 = tuple(rs[:5])
        return (5, top5)

    # Стрит?
    st_hi = highest_straight_from_mask(rank_mask)
    if st_hi != -1:
        return (4, (st_hi,))

    # Тройка?
    trips = -1
    for r in range(12, -1, -1):
        if rank_counts[r] == 3:
            trips = r
            break
    if trips != -1:
        kickers = []
        for r in range(12, -1, -1):
            if r != trips and rank_counts[r] > 0:
                kickers.append(r)
                if len(kickers) == 2:
                    break
        return (3, (trips, kickers[0], kickers[1]))

    # Две пары?
    pair_ranks = []
    for r in range(12, -1, -1):
        if rank_counts[r] >= 2:
            pair_ranks.append(r)
            if len(pair_ranks) == 2:
                break
    if len(pair_ranks) >= 2:
        kicker = -1
        for r in range(12, -1, -1):
            if r not in pair_ranks and rank_counts[r] > 0:
                kicker = r
                break
        return (2, (pair_ranks[0], pair_ranks[1], kicker))

    # Одна пара?
    pair_rank = -1
    for r in range(12, -1, -1):
        if rank_counts[r] == 2:
            pair_rank = r
            break
    if pair_rank != -1:
        kickers = []
        for r in range(12, -1, -1):
            if r != pair_rank and rank_counts[r] > 0:
                kickers.append(r)
                if len(kickers) == 3:
                    break
        return (1, (pair_rank, kickers[0], kickers[1], kickers[2]))

    # Старшая карта
    highs = []
    for r in range(12, -1, -1):
        highs.extend([r] * rank_counts[r])
    return (0, tuple(highs[:5]))


# ------------------------------------------------------------
# Модель игрока и выбор рук
# ------------------------------------------------------------

@dataclass
class PlayerSpec:
    name: str
    range_spec: RangeSpec


def has_conflicts(cards: Sequence[int]) -> Optional[str]:
    """Проверка на дубликаты карт (между всеми наборами). Возвращает сообщение об ошибке или None."""
    s = set()
    dups = []
    for c in cards:
        if c in s:
            dups.append(int_to_card(c))
        s.add(c)
    if dups:
        return "Дубликаты карт: " + ", ".join(sorted(dups))
    return None


def sample_random_hand_from_deck(deck: List[int], rng: random.Random) -> Tuple[int, int]:
    """Случайная рука (2 карты) из текущей колоды."""
    c1, c2 = rng.sample(deck, 2)
    if c2 < c1:
        c1, c2 = c2, c1
    return (c1, c2)


def sample_from_range(range_spec: RangeSpec, deck: List[int], rng: random.Random,
                      max_attempts: int = 200) -> Optional[Tuple[int, int]]:
    """
    Выбор одной комбо из диапазона без коллизий с текущей колодой.
    Возвращает (c1,c2) или None, если невозможно.
    """
    if range_spec.is_random:
        # Случайная рука из оставшихся карт
        if len(deck) < 2:
            return None
        return sample_random_hand_from_deck(deck, rng)

    combos = range_spec.combos or []
    if not combos:
        return None

    # Быстрый rejection sampling
    n = len(combos)
    for _ in range(max_attempts):
        c1, c2 = combos[rng.randrange(n)]
        if c1 in deck and c2 in deck:
            return (c1, c2)

    # Фолбэк: фильтрация списка
    feasible = [(a, b) for (a, b) in combos if (a in deck and b in deck)]
    if not feasible:
        return None
    return feasible[rng.randrange(len(feasible))]


# ------------------------------------------------------------
# Симуляция Монте-Карло и точный перебор
# ------------------------------------------------------------

@dataclass
class SimContext:
    players: List[PlayerSpec]
    board_fixed: List[int]
    dead_cards: List[int]
    iters: int
    seed: Optional[int]
    progress: bool
    timeout_sec: Optional[float]
    mode: str  # 'montecarlo' or 'exact'
    max_exact: int


@dataclass
class PlayerCounters:
    name: str
    range_label: str
    wins: int = 0
    ties: int = 0
    share: float = 0.0  # сумма долей банка: 1 при соло-победе, 1/k при делёжке

    def as_dict(self, total_iters: int) -> Dict:
        equity = self.share / max(1, total_iters)
        losses = total_iters - self.wins - self.ties
        return {
            "name": self.name,
            "range": self.range_label,
            "equity": round(equity, 6),
            "wins": self.wins,
            "ties": self.ties,
            "losses": losses,
        }


def complete_board(current_board: List[int], deck: List[int], rng: random.Random) -> List[int]:
    """Дороздача недостающих общих карт до 5."""
    need = 5 - len(current_board)
    if need <= 0:
        return list(current_board)
    draw = rng.sample(deck, need)
    return current_board + draw


def simulate_monte_carlo(ctx: SimContext) -> Tuple[List[PlayerCounters], int, float]:
    """Основной цикл Монте-Карло. Возвращает (счётчики, выполненные_итерации, runtime_sec)."""
    rng = random.Random(ctx.seed)
    players = ctx.players
    counters = [PlayerCounters(p.name, p.range_spec.label) for p in players]

    start = time.perf_counter()
    last_print = start
    done = 0

    # Для удобства: предвычтем конфликтующие с board/dead карты из колоды на каждой итерации
    base_dead = set(ctx.board_fixed + ctx.dead_cards)
    # Проверка: ни одна фиксированная рука не должна конфликтовать с уже указанной доской/мертвыми картами
    for p in players:
        if not p.range_spec.is_random and p.range_spec.combos:
            # Если диапазон — это ровно одна точная рука, и она конфликтует, упадём сразу
            if len(p.range_spec.combos) == 1:
                (a, b) = p.range_spec.combos[0]
                if a in base_dead or b in base_dead:
                    raise ValueError(f"Рука игрока {p.name} конфликтует с board/dead.")
            # Также проверим, что из диапазона после учёта dead остались хоть какие-то комбо
            feasible = 0
            for (a, b) in p.range_spec.combos:
                if a not in base_dead and b not in base_dead:
                    feasible += 1
            if feasible == 0:
                raise ValueError(f"Диапазон игрока {p.name} заблокирован мёртвыми/бордом.")

    while done < ctx.iters:
        now = time.perf_counter()
        if ctx.timeout_sec is not None and (now - start) >= ctx.timeout_sec:
            break

        # Сформируем свежую колоду для итерации
        deck = deck52()
        remove_cards(deck, ctx.board_fixed)
        remove_cards(deck, ctx.dead_cards)

        # Выберем руки по порядку: сначала герой, затем оппы
        chosen_hands: List[Tuple[int, int]] = []
        for p in players:
            hand = sample_from_range(p.range_spec, deck, rng)
            if hand is None:
                # Невозможно вытащить руку из диапазона -> пропускаем итерацию
                # (в норме этого быть не должно, т.к. мы проверяем feasibility заранее)
                break
            a, b = hand
            # Уберём карты из колоды
            deck.remove(a)
            deck.remove(b)
            chosen_hands.append(hand)
        else:
            # Дороздаём борд
            board = complete_board(ctx.board_fixed, deck, rng)

            # Оценим силы рук
            ranks = []
            for (a, b) in chosen_hands:
                ranks.append(hand_rank_7([a, b] + board))

            # Определим победителя(ей)
            best = max(ranks)
            winners = [i for i, r in enumerate(ranks) if r == best]
            k = len(winners)
            if k == 1:
                w = winners[0]
                counters[w].wins += 1
                counters[w].share += 1.0
            else:
                for w in winners:
                    counters[w].ties += 1
                    counters[w].share += 1.0 / k

            done += 1

            # Прогресс-бар
            if ctx.progress:
                if now - last_print >= 0.25:
                    rate = done / max(1e-9, now - start)
                    pct = 100.0 * done / ctx.iters
                    eta = (ctx.iters - done) / rate if rate > 0 else float('inf')
                    sys.stderr.write(
                        f"\r{pct:5.1f}% | {done:_d}/{ctx.iters:_d} | {rate/1e6:,.2f} M it/s | ETA ~ {eta:5.1f}s"
                    )
                    sys.stderr.flush()
                    last_print = now

    if ctx.progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    runtime = time.perf_counter() - start
    return counters, done, runtime


def simulate_exact(ctx: SimContext) -> Tuple[List[PlayerCounters], int, float]:
    """
    Точный перебор реализован для случая, когда У ВСЕХ игроков диапазон сводится
    к РОВНО ОДНОЙ фиксированной руке (по одной комбо на игрока).
    В остальных случаях — возврат с вежливым отказом (будет обработан вне).
    """
    players = ctx.players
    # Проверим условие фиксированных рук
    fixed_hands: List[Tuple[int, int]] = []
    for p in players:
        rs = p.range_spec
        if rs.is_random:
            raise ValueError("Exact недоступен: у одного из игроков 'random'.")
        if not rs.combos or len(rs.combos) != 1:
            raise ValueError("Exact доступен только при фиксированных руках для всех игроков.")
        fixed_hands.append(rs.combos[0])

    # Проверим конфликты с board/dead
    used = list(ctx.board_fixed + ctx.dead_cards)
    for (a, b) in fixed_hands:
        used.extend([a, b])
    msg = has_conflicts(used)
    if msg:
        raise ValueError(f"Конфликт карт: {msg}")

    # Перебор недостающих карт борда
    base_deck = deck52()
    remove_cards(base_deck, used)
    need = 5 - len(ctx.board_fixed)
    total_states = comb(len(base_deck), need)
    if total_states > ctx.max_exact:
        raise ValueError(
            f"Exact слишком большой: C({len(base_deck)},{need}) = {total_states:_d} > --max-exact={ctx.max_exact:_d}."
        )

    counters = [PlayerCounters(p.name, p.range_spec.label) for p in players]
    start = time.perf_counter()
    last_print = start
    done = 0

    for add_board in itertools.combinations(base_deck, need):
        board = ctx.board_fixed + list(add_board)
        ranks = []
        for (a, b) in fixed_hands:
            ranks.append(hand_rank_7([a, b] + board))
        best = max(ranks)
        winners = [i for i, r in enumerate(ranks) if r == best]
        k = len(winners)
        if k == 1:
            w = winners[0]
            counters[w].wins += 1
            counters[w].share += 1.0
        else:
            for w in winners:
                counters[w].ties += 1
                counters[w].share += 1.0 / k
        done += 1

        # Лёгкий прогресс для exact
        now = time.perf_counter()
        if ctx.progress and (now - last_print >= 0.25):
            pct = 100.0 * done / total_states if total_states else 100.0
            rate = done / max(1e-9, now - start)
            eta = (total_states - done) / rate if rate > 0 else float('inf')
            sys.stderr.write(
                f"\r{pct:5.1f}% | {done:_d}/{total_states:_d} | {rate/1e6:,.2f} M it/s | ETA ~ {eta:5.1f}s"
            )
            sys.stderr.flush()
            last_print = now

    if ctx.progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    runtime = time.perf_counter() - start
    return counters, done, runtime


# ------------------------------------------------------------
# Форматирование вывода
# ------------------------------------------------------------

def stderr(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def format_human_output(mode: str, players: List[PlayerSpec], counters: List[PlayerCounters],
                        board: List[int], iters: int, runtime: float, seed: Optional[int]) -> str:
    lines = []
    board_str = "".join(int_to_card(c) for c in board)
    header = (
        f"Poker Equity Calculator ({'Exact' if mode=='exact' else 'Monte Carlo'})\n"
        f"Players: {len(players)} | Board: {board_str or '-'} | Iters: {iters:_d} | "
        f"Seed: {seed if seed is not None else '-'} | Time: {runtime:.2f}s"
    )
    lines.append(header)
    lines.append("-" * len(header))

    # Подготовим столбцы
    name_w = max(4, max(len(p.name) for p in players))
    range_w = max(5, max(len(p.range_spec.label) for p in players))
    for i, cnt in enumerate(counters):
        eq = 100.0 * cnt.share / max(1, iters)
        losses = iters - cnt.wins - cnt.ties
        # Стандартная ошибка (приближение Бернулли по equity)
        p_hat = cnt.share / max(1, iters)
        stderr_pct = 100.0 * math.sqrt(max(0.0, p_hat * (1 - p_hat) / max(1, iters)))
        lines.append(
            f"{cnt.name:<{name_w}} ({cnt.range_label:<{range_w}}):  "
            f"Equity {eq:6.2f}% | Wins {cnt.wins:_d} | Ties {cnt.ties:_d} | "
            f"Losses {losses:_d} | StdErr ±{stderr_pct:.2f}%"
        )

    return "\n".join(lines)


def format_json_output(mode: str, players: List[PlayerSpec], counters: List[PlayerCounters],
                       board: List[int], iters: int, runtime: float,
                       seed: Optional[int]) -> str:
    obj = {
        "mode": "exact" if mode == "exact" else "montecarlo",
        "iterations": iters,
        "players": [c.as_dict(iters) for c in counters],
        "board": "".join(int_to_card(c) for c in board),
        "runtime_sec": round(runtime, 6),
        "seed": seed
    }
    return json.dumps(obj, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# CLI и main
# ------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Калькулятор equity для Техасского Холдема (один файл, без внешних зависимостей).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--hero", required=True, help="Рука/диапазон героя (напр., AsKs или 'JJ+,AKs,AQo+')")

    # Способы задать оппонентов
    p.add_argument("--villain", action="append", help="Рука/диапазон одного оппонента (можно повторять).")
    p.add_argument("--villains", nargs="*", help="Список рук/диапазонов оппонентов.")
    p.add_argument("--players", type=int, default=None,
                   help="Число игроков (2–6), если все оппоненты случайные ('random').")

    p.add_argument("--board", default="", help="Общие карты (0–5), например QsJh2h или QsJh2hTd9c.")
    p.add_argument("--dead", default="", help="Мёртвые карты (не участвуют в раздаче).")

    p.add_argument("--iters", type=int, default=200_000, help="Число итераций Монте-Карло.")
    p.add_argument("--mode", choices=["montecarlo", "exact"], default="montecarlo", help="Алгоритм расчёта.")
    p.add_argument("--max-exact", type=int, default=3_000_000, help="Максимальный объём перебора для exact.")

    p.add_argument("--json", action="store_true", help="Вывести результаты в JSON.")
    p.add_argument("--progress", action="store_true", help="Показывать прогресс.")
    p.add_argument("--seed", type=int, default=None, help="Seed для генератора случайных чисел.")
    p.add_argument("--timeout-sec", type=float, default=None, help="Мягкий таймаут (сек), по истечении — остановка.")

    p.add_argument("--verbose", action="store_true", help="Подробные сообщения (парсинг диапазонов и т.п.).")
    p.add_argument("--selftest", action="store_true", help="Запустить встроенные проверки и выйти.")
    return p


# ------------------------------------------------------------
# Самопроверка (минимальный набор регрессионных тестов)
# ------------------------------------------------------------

def self_test() -> None:
    # Тест карт
    assert int_to_card(card_to_int("As")) == "As"
    assert int_to_card(card_to_int("Td")) == "Td"
    try:
        card_to_int("ZZ")
        raise AssertionError("Ожидалась ошибка на неверной карте.")
    except ValueError:
        pass

    # Тест диапазонов
    assert try_parse_exact_hand("AsKs") == tuple(sorted((card_to_int("As"), card_to_int("Ks"))))
    rs = parse_range("JJ+")
    # В JJ+ должно быть пары JJ,QQ,KK,AA по 6 комбо -> 24
    assert not rs.is_random and rs.combos and len(rs.combos) == 4 * 6

    rs2 = parse_range("ATs+")
    # ATs+, AJs, AQs, AKs -> 4 * 4 = 16 комбо
    assert not rs2.is_random and rs2.combos and len(rs2.combos) == 16

    rs3 = parse_range("22-99")
    assert len(rs3.combos) == (8 * 6)  # пары от 22..99

    # Тест ранжирования рук (несколько известных сравнений)
    # Стрит-флеш > каре
    sflush = [card_to_int(c) for c in ("Ah", "Kh", "Qh", "Jh", "Th", "2c", "3d")]
    quads = [card_to_int(c) for c in ("9c", "9d", "9h", "9s", "2c", "3d", "4h")]
    assert hand_rank_7(sflush)[0] == 8
    assert hand_rank_7(quads)[0] == 7
    assert hand_rank_7(sflush) > hand_rank_7(quads)

    # Сет 7х на доске 5h6c7s8d9h
    seven_set = [card_to_int(c) for c in ("7h", "7d", "5h", "6c", "7s", "8d", "9h")]
    assert hand_rank_7(seven_set)[0] >= 3

    # Wheel A-5 (A2345) как стрит
    wheel = [card_to_int(c) for c in ("Ah", "2d", "3c", "4h", "5s", "Td", "9c")]
    assert hand_rank_7(wheel)[0] >= 4

    print("Self-test OK.")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.selftest:
        self_test()
        return

    # Соберём список оппонентов
    villains: List[str] = []
    if args.villain:
        villains.extend(args.villain)
    if args.villains:
        villains.extend(args.villains)
    if args.players is not None:
        if villains:
            parser.error("Нельзя одновременно использовать --players и --villain/--villains.")
        if args.players < 2 or args.players > 6:
            parser.error("--players должен быть в диапазоне 2..6.")
        # Все оппоненты случайные
        villains = ["random"] * (args.players - 1)
    if not villains:
        parser.error("Не указаны оппоненты. Используйте --villain/--villains или --players.")

    # Парсим board и dead
    try:
        board_fixed = parse_cards_concat(args.board)
        dead_cards = parse_cards_concat(args.dead)
    except ValueError as e:
        parser.error(str(e))

    # Проверим конфликты в board/dead
    msg = has_conflicts(board_fixed + dead_cards)
    if msg:
        parser.error(msg)

    # Парсим диапазоны
    try:
        hero_rs = parse_range(args.hero)
        villain_rs = [parse_range(v) for v in villains]
    except ValueError as e:
        parser.error(str(e))

    players = [PlayerSpec("Hero", hero_rs)]
    for i, rs in enumerate(villain_rs, start=1):
        players.append(PlayerSpec(f"V{i}", rs))

    if args.verbose:
        stderr("Разобранные диапазоны:")
        for p in players:
            if p.range_spec.is_random:
                stderr(f"  {p.name}: random")
            else:
                cnt = len(p.range_spec.combos or [])
                stderr(f"  {p.name}: {p.range_spec.label} -> {cnt} комбо")

    # Построим контекст симуляции
    ctx = SimContext(
        players=players,
        board_fixed=board_fixed,
        dead_cards=dead_cards,
        iters=max(1, args.iters),
        seed=args.seed,
        progress=args.progress,
        timeout_sec=args.timeout_sec,
        mode=args.mode,
        max_exact=max(1, args.max_exact),
    )

    # Запуск
    try:
        if ctx.mode == "exact":
            try:
                counters, done, runtime = simulate_exact(ctx)
                out_board = ctx.board_fixed  # доска фиксирована в exact + дороздана в переборе
            except Exception as e:
                # Вежливо сообщаем и откатываемся к Монте-Карло
                stderr(f"[Exact] {e}\nПерехожу в режим Monte Carlo…")
                ctx.mode = "montecarlo"
                counters, done, runtime = simulate_monte_carlo(ctx)
                out_board = ctx.board_fixed
        else:
            counters, done, runtime = simulate_monte_carlo(ctx)
            out_board = ctx.board_fixed

        # Вывод
        if args.json:
            print(format_json_output(ctx.mode, ctx.players, counters, out_board, done, runtime, ctx.seed))
        else:
            print(format_human_output(ctx.mode, ctx.players, counters, out_board, done, runtime, ctx.seed))

    except KeyboardInterrupt:
        stderr("\nПрервано пользователем (Ctrl+C).")
        sys.exit(1)
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()