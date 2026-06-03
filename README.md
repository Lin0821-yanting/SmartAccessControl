
## Jetson 環境軟連結設定

由於 Jetson 系統套件需透過軟連結引入 venv，執行以下指令：

```bash
VENV_SITE="/home/jetson/SmartAccessControl/.venv/lib/python3.10/site-packages"
SYS1="/usr/local/lib/python3.10/dist-packages"
SYS2="/usr/lib/python3/dist-packages"
LOCAL="/home/jetson/.local/lib/python3.10/site-packages"

for pkg in mpmath sympy contourpy dateutil fonttools pandas scipy seaborn tqdm requests ultralytics ultralytics_thop torch torchvision torchgen PIL numpy numpy.libs matplotlib mpl_toolkits charset_normalizer; do
    [ -e $SYS1/$pkg ] && ln -sf $SYS1/$pkg $VENV_SITE/$pkg
done
for pkg in urllib3 idna certifi; do
    [ -e $SYS2/$pkg ] && ln -sf $SYS2/$pkg $VENV_SITE/$pkg
done
ln -sf $SYS2/six.py $VENV_SITE/six.py
ln -sf $SYS2/pyparsing.py $VENV_SITE/pyparsing.py
for pkg in certifi cycler kiwisolver; do
    [ -e $LOCAL/$pkg ] && ln -sf $LOCAL/$pkg $VENV_SITE/$pkg
done
```
