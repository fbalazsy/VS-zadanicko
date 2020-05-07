# Druhá úloha - Houghova transformácia

## Spustenie programu

Zadanie bolo vypracovane v jazyku Python, v3, takže tento je potrebné mať. Boli použité knižnice:

**numpy** a **matplotlib**, takže pred spustením je potrebné ich nainštalovať:

```
pip install numpy matplotlib
```
Spustenie, napr.:
```
#meno skriptu <stupny_obraz> <prah_pre_vyber_ciar>

hough_transform.py test_images/sudoku.jpg 150
```

## Vstupné obrázky
Pre testovanie sme použili tri rôzne obrázky, ktoré sú v priečinku `test_images`,
- jednoducký, s ľahko identifikovateľnými hranami

![image info](./test_images/pentagon.jpg)

- trochu komplikovanejší, ale stále s jasnými hranami

![image info](./test_images/sudoku.jpg)

- komplikovaný, ale ešte s viditeľnými hranami 

![image info](./test_images/Window.jpg)

## Predspracovanie

Pre úspešné použitie *houghovej transformácie* je potrebné dostať obraz do formy, kde sú zdetekované hrany. Za týmto účelom sme použili **Cannyho hranový detektor**, ktorý pozostáva zo štyroch krokov: 
1. Aplikovanie *Gausovho filtra* na odstránenie šumu
2. Nájdenie intezity grandietu a smeru hrany
3. Stenšenie hrán pomocou odobratia bodov, ktoré nie sú lokálnymi maximami
4. Aplikovanie prahovania na zachovanie bodov s vysokým gradietom

Najvýraznejší vplyv malo vhodne zvolené prahovanie, a to také, v ktorom sa bral ohľad na komplexnosť obrázka, pri jednoduchom sme volili vysoký prah, no a čím bol obrázok komplikovanejší tým bolo lepšie aj voliť nižšie prahy.

  Dostali sme takéto výsledky:

![image info](./edge_detected_images/Pentagon_edges.png)

- použili sme 75%-ný prah (75% max. hodnoty v obrázku)

![image info](./edge_detected_images/Sudoku_edges.png)

- použili sme 10%-ný prah, pri vyšších sme prichádzali o niektoré hrany

![image info](./edge_detected_images/Window_edges.png)

- použili sme 30%-ný prah, tu sme zaznamenali vysokú náchylnosť na voľbu tohto prahu, keďže ide o komplikovaný obrázok, 30 bol výsledný kompromis.

## Aplikovanie Hougovej transformacie

Po pripravení obrázkov sme mohli pristúpiť k samotnej **Houghovej transformácií**, pomocou ktorej sme detekovali čiary, už relatívne jednoduchým postupom, pri ktorom využívame, že priamky môžme reprezentovať nasledovnou formulou:

```
ro = x * cos(theta) + y * sin(theta)
```
kde **x** a **y** sú koordináty na obrázku.
Tým pádom vieme akúkoľvek priamku reprezentovať ako kombináciu dvoch parametrov **theta** a **ro**.
Vytvorili sme z nich dvojrozmerné pole, kde na x-ovej osi sme naniesli hodnoty **theta**, s možnými hodnotami 0, 1, ..., 180
a na y-ovú os sme naniesli hodnoty **ro**, ktorého rozsah hodnôt závisí od veľkosti vstupného obrazká, 
keďže to je v podstate vzdialenosť k definovaj priamke od rohu obrázka, (**theta** je uhol, ktorý zviera **ro** s definovanou priamkou).

Keď už máme definovane naše pole rôznych kombinácií **theta** a **ro** vytiahneme si z nášho predspracovaného obrázka hrany s určitou intenzitou.
(použili sme hodnotu **5**) a do poľa **theta** a **ro** (akumulátora) zvyšujeme hodnoty na miestach, ktoré intenzívnych priamkam (intenzita > 5) odpovedajú

Keď sme už vytvorili akumulátor, ľahko z neho vytiahneho získame hľadané priamky, aplikovaním prahu, ktorý opäť je špecifický pre každý obrázok, 
Vykreslenie spočíva už len v upravení vyššie definovanej rovnice, do tejto podoby:
```
y = (ro - x * cos(theta)) / sin(theta)
```

Dosiahli sme nasledovné výsledky:
- prah pre výber z akumulátora - **100**
![image info](./output_images/Hough_pentagon.png)
![image info](./output_images/Pentagon_final.png)


- prah pre výber z akumulátora - **150**
![image info](./output_images/Hough_Sudoku.png)
![image info](./output_images/Sudoku_final.png)


- prah pre výber z akumulátora - **80**
![image info](./output_images/Hough_Window.png)
![image info](./output_images/Window_final.png)



