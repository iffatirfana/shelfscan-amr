# Submission checklist (manual steps)

## 1. Push to GitHub
`gh` (GitHub CLI) is not installed on this machine yet.

```powershell
winget install --id GitHub.cli        # then open a NEW terminal
gh auth login                         # GitHub.com -> HTTPS -> login with browser
cd D:\ihacks\shelfscan-amr
gh repo create shelfscan-amr --public --source . --remote origin --push
```

Without gh: create an empty repo on github.com (no README), then
```powershell
git remote add origin https://github.com/<your-user>/shelfscan-amr.git
git push -u origin main
```

## 2. Fill in links, rebuild the deck/PDF, commit, push
Edit the three constants at the top of `docs/build_submission.py`
(`VIDEO_URL`, `GITHUB_URL`, `TEAM`) and the video line in `README.md`, then:
```powershell
python docs/build_submission.py
git commit -am "Add video and repo links"; git push
```

## 3. Record the video (script: docs/video_script.md)
Screen-record one audit in the Webots GUI (realtime), then the slides.

## 4. Upload
- `docs/ShelfScan_submission.pdf` = deck + 2-3 page concept note in ONE PDF
- demo video link, GitHub link
