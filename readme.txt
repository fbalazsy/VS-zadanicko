1. Uloha - pravitko-vizs

#Nainstalujeme potrebne baliky
pip install numpy opencv-python
#Spustime skript ulohy
python prod.py

V prvej časti úlohy bolo pre správne meranie vzdialenosti nevyhnutné urobiť kalibráciu skreslenia kamery.Počas tvorby meracieho algoritmu sme vystriedali viaceré prístupy ako napríklad detekovať hrany na celom zábere pomocou Houghovej transformácie a počítať šírku z nich. Najviac sa nám však osvedčilo striktne obmedziť skenovanú oblasť, využiť podmienky úlohy(biela látka na čiernom dopravnom páse) a nájsť biele oblasti jednoduchým treshholdom. Následne si nájdeme kontúry oblastí a vytriedime tie, ktoré spĺňajú limity veľkosti plochy. Meranie prebieha v oblasti záberu najviac vyhovujúcim podmienkam zadania, to tak že na danom mieste sa látka musí nachádzať vždy po šírke celá(bez počiatočných rožkov). Poloha a natočenie kamery je fixne dané, vzťah medzi skutočnou dĺžkou a počtom pixelov z linearizovaného obrazu sme definovali meraním.
