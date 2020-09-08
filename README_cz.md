# Automatickcé vyhodnocování slovních úloh
Tento projekt vznikl v rámci bakalářské práce.
Cílem tohoto projektu bylo vytvořit program, který by 
řešil jednoduché matematické slovní úlohy v českém jazyce pro žáky 1. - 3. třídy základních škol.

Tento program používá extrahovaná data z korpusu SYN2015.  
https://wiki.korpus.cz/doku.php/:cnk:syn2015

Potřebné knihovny pro spuštění jsou specifikovány v souboru requirements.txt.

# Spuštění
Bohužel licence slovních úloh nám neumožňuje zveřejnění. A proto jsou adresáře dataset a data/traindata prázdné.

## Dataset
Bylo použito 500 slovních úloh v českém jazyce z učebnice [[1]](#1).
Byly vytvořeny dva datasety:
WP150 - obsahující 150 slovních úloh
WP500 - obsahující 500 slovních úloh

Příklad slovní úlohy:
```bash
Maminka koupila na trhu 15 kg třešní a 3 krát méně jahod. Kolik kg ovoce koupila dohromady? | 20 | NUM1 + NUM1 / NUM2
```