if [ -L ./baseball/dp ]; then
    unlink baseball/dp
fi
if [ -L ./volleyball/dp ]; then
    unlink volleyball/dp
fi
if [ -L ./gui/dp ]; then
    unlink gui/dp
fi

ln -s $(pwd)/dp baseball/
ln -s $(pwd)/dp volleyball/
ln -s $(pwd)/dp gui/
