# based on https://stackoverflow.com/a/40178818/11051330

STATUS="$(git status)"
DOCSDIR="docs/_build/"

if [[ $STATUS == *"nothing to commit, working tree clean"* ]]
then
    sed -i "" '$DOCSDIR' ./.gitignore
    git add .
    git commit -m "Edit .gitignore to publish docs"
    git subtree push --prefix $DOCSDIR/html/ origin gh-pages
    git reset HEAD~
    git checkout .gitignore
else
    echo "Need clean working directory to publish"
fi
