# 1. move the tests you forgot
mkdir -p tests/bicep
git mv backends/bicep/tests/* tests/bicep/ 2>/dev/null || true   # if they exist

# 2. make fusion_alpha sub-packages
for p in models pipelines routers; do
  mkdir -p fusion_alpha/$p
  touch fusion_alpha/$p/__init__.py
done
git add fusion_alpha/{models,pipelines,routers}/__init__.py

# 3. housekeeping
echo -e "artifacts/\n__pycache__/\n*.pth\n*.db\n" >> .gitignore
git add .gitignore
git commit -m "package tweaks: sub-modules + gitignore"
