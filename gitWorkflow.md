# Github Workflow Overview

## Downloading Old Work
1. Before you start working, it's probably helpful to pull from the server. This is done with: 
  - `git pull`
2. If you need to re-clone from the github, this is your command. I've put the github for this project there: 
  - `git clone https://github.com/ruetten/cosmic_sleuth`
  - `ls` - this will just list the files you downloaded


## Uploading New Work
1. You'll want to start by pulling any work quickly by doing (to make sure you don't overwrite other people's work):
  - `git pull`
2. Next you'll want to say all the files you want. You can add all files in a folder with `*` or specify filenames. 
  - `git add *`
  - `git add [filename1] [filename2] ... `
3. Next you'll want to commit with a helpful message of what you just changed
  - `git commit -m 'Write your helpful message here in quotes'`
4. [Optionally] You can make sure you added all the files with status to make sure you added everything
  - `git status`
5. Now you'll want to push to the remote server. You can specify which branch to push on (probably `main`, which is the main branch)
  - `git push`
  - `git push origin main`

## Branching Stuff (if you're curious)
1. Here's some notes about branching (you don't really need to worry about it). We'll start with making a branch. `branch` makes a branch, with the given name, `checkout` switches you to the named branch, then you can do all the git commands on the named branch

```
git pull
git branch branch1
git branch 
git checkout branch1
echo "yet another file" >> thirdfile.txt
git add thirdfile.txt
git status
git commit -m "I made a branch" 
git checkout main
git push 
```

2. You can also try to merge branches with the following command:

```
git branch
git merge b1
git branch
git commit -m "Merged the branches"
```

3. Lastly, writing `log` will show you a history of changes to the github: 
  - `git log`