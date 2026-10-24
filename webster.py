"""Расчёт фиксированного плана по методу Вебстера для нашего перекрёстка.

Допущения (их надо описать в главе 2):
  * поток насыщения s = 3600 маш/ч на направление (две полосы);
  * потерянное время L = 8 с за цикл (два жёлтых по 2 с + по 2 с на разгон);
  * цикл ограничен диапазоном [30, 120] с, минимальный зелёный 7 с.
"""


def webster_plan(ns, we, s=3600.0, lost=8.0, c_min=30.0, c_max=120.0, g_min=7):
    """ns, we: вероятность появления машины в секунду (север-юг, запад-восток).
    Возвращает (g_ns, g_we, C): зелёные в секундах и длину цикла."""
    y_ns = ns * 3600.0 / s
    y_we = we * 3600.0 / s
    Y = y_ns + y_we
    if Y >= 0.95:                       # почти перегруз: формула расходится
        C = c_max
    else:
        C = min(c_max, max(c_min, (1.5 * lost + 5.0) / (1.0 - Y)))
    g_total = C - lost
    g_ns = max(g_min, round(g_total * y_ns / Y))
    g_we = max(g_min, round(g_total * y_we / Y))
    return g_ns, g_we, C


def webster_tl_xml(g_ns, g_we):
    """Блок <tlLogic> со статической программой (фазы как в исходной сети)."""
    return (
        '<tlLogic id="t" type="static" programID="0" offset="0">\n'
        f'        <phase duration="{g_ns}" state="GGrr"/>\n'
        '        <phase duration="2" state="yyrr"/>\n'
        f'        <phase duration="{g_we}" state="rrGG"/>\n'
        '        <phase duration="2" state="rryy"/>\n'
        '    </tlLogic>'
    )


if __name__ == "__main__":
    loads = {"low": (0.10, 0.25), "medium": (0.20, 0.50),
             "high": (0.26, 0.65), "symmetric": (0.35, 0.35)}
    for name, (ns, we) in loads.items():
        print(name, webster_plan(ns, we))
