import math
from statistics import NormalDist


# Квантиль стандартного нормального распределения
z = NormalDist().inv_cdf


def n_per_group(
    sigma: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    comparisons: int = 1,
) -> int:
    """Сколько клиентов нужно на каждую группу."""
    
    # Учитываем уровень значимости и число сравнений
    za = z(1 - alpha / (2 * comparisons))

    # Квантиль, отвечающий за мощность
    zb = z(power)

    # Считаем размер каждой группы и округляем вверх
    return math.ceil(
        2 * (za + zb) ** 2 * sigma ** 2 / mde ** 2
    )


def mde_for_n(
    sigma: float,
    n: int,
    alpha: float = 0.05,
    power: float = 0.8,
    comparisons: int = 1,
) -> float:
    """Какой минимальный эффект заметит эксперимент."""
    
    # Те же квантили, что в формуле размера выборки
    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)

    # Обратная формула к n_per_group
    return (
        2 * (za + zb) ** 2 * sigma ** 2 / n
    ) ** 0.5


def diff_ci(t, c, alpha: float = 0.05) -> dict:
    """Разность средних test − control, интервал и p-значение."""
    
    # Эффект: среднее теста минус среднее контроля
    d = t.mean() - c.mean()

    # Стандартная ошибка разности средних
    se = (
        t.var() / len(t)
        + c.var() / len(c)
    ) ** 0.5

    # Граница для доверительного интервала
    q = NormalDist().inv_cdf(1 - alpha / 2)

    # Двустороннее p-значение
    p = 2 * (1 - NormalDist().cdf(abs(d / se)))

    return {
        "diff": d,
        "lo": d - q * se,
        "hi": d + q * se,
        "p": p,
    }


def srm_pvalue(sizes) -> float:
    """p-значение SRM для двух или трёх равных групп."""
    
    # Общее число клиентов
    n = sum(sizes)

    # Сколько ожидалось бы в каждой группе
    exp = n / len(sizes)

    # Статистика chi-square
    chi2 = sum(
        (s - exp) ** 2 / exp
        for s in sizes
    )

    # Формула зависит от числа групп
    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi2 / 2))

    if len(sizes) == 3:
        return math.exp(-chi2 / 2)

    raise ValueError(
        "для 4 и более групп — scipy.stats.chisquare"
    )


def cuped(y, x):
    """Метрика после CUPED и коэффициент theta."""
    
    # Насколько итоговая метрика связана с метрикой до эксперимента
    theta = y.cov(x) / x.var()

    # Убираем из y предсказуемую по прошлому часть
    y_cuped = y - theta * (x - x.mean())

    return y_cuped, theta