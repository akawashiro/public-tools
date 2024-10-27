#!/bin/bash -eux

ghq get git@github.com:akawashiro/julia.git
cd $(ghq root)/github.com/akawashiro/julia
git remote add upstream git@github.com:JuliaLang/julia.git
git remote set-url upstream --push no-push
