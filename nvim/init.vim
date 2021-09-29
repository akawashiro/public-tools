if !&compatible
  set nocompatible
endif

" reset augroup
augroup MyAutoCmd
  autocmd!
augroup END

" ========== dein settings start ==========
" Auto install of dein itself
let s:cache_home = empty($XDG_CACHE_HOME) ? expand('~/.cache') : $XDG_CACHE_HOME
let s:dein_dir = s:cache_home . '/dein'
let s:dein_repo_dir = s:dein_dir . '/repos/github.com/Shougo/dein.vim'
if !isdirectory(s:dein_repo_dir)
  call system('git clone https://github.com/Shougo/dein.vim ' . shellescape(s:dein_repo_dir))
endif
let &runtimepath = s:dein_repo_dir .",". &runtimepath

" Load plugins and make cache
let s:toml_file = fnamemodify(expand('<sfile>'), ':h').'/dein.toml'
if dein#load_state(s:dein_dir)
  call dein#begin(s:dein_dir)
  call dein#load_toml(s:toml_file)
  call dein#end()
  call dein#save_state()
endif

" Auto install of uninstalled plugins
if has('vim_starting') && dein#check_install()
  call dein#install()
endif

" ========== dein settings end ==========

" ========== nvim start ==========

" Use newer python installed in home directory.
let g:python3_host_prog = substitute(system('which python3'),"\n","","")

filetype plugin indent on
syntax enable
set number
set shiftwidth=4
set tabstop=4
set expandtab
set cindent
syntax on

colorscheme darkblue

set spell
highlight clear SpellBad
highlight SpellBad cterm=underline
" Set style for gVim
highlight SpellBad gui=undercurl

set nocompatible
set whichwrap=b,s,h,l,<,>,[,],~
set backspace=indent,eol,start

set clipboard+=unnamedplus
set hlsearch
nmap <Esc><Esc> :nohlsearch<CR><Esc>
inoremap jj <Esc>

set guifont =DejaVu\ Sans\ Mono\ 10 
set nofoldenable    " disable folding

set cursorcolumn

" buffer
map <silent> tn :bnext<CR>
map <silent> tp :bprevious<CR>
map <silent> td :bdelete<CR>

" Without this option, you cannot switch between tabs before saving.
set hidden

set completeopt=menuone

" ========== nvim end ==========

" ========== deoplete start ==========

let g:deoplete#enable_at_startup = 1

" ========== deoplete end ==========

" ========== fzf start ==========

" 'options': '--delimiter : --nth 3..' exist not to match with filenames

command! -bang -nargs=* GGrep
  \ call fzf#vim#grep(
  \   'git grep --line-number -- '.shellescape(<q-args>), 0,
  \   fzf#vim#with_preview({'dir': systemlist('git rev-parse --show-toplevel')[0], 'options': '--delimiter : --nth 3..'}), <bang>0)
command! -bang -nargs=* Ag call fzf#vim#ag(<q-args>, {'options': '--delimiter : --nth 3..'}, <bang>0)

" ========== fzf end ==========

" ========== julia start ==========

autocmd BufNewFile,BufRead *.jl setfiletype julia

" ========== rhubarb.vim start ===========

let g:github_enterprise_urls = ['https://github.pfidev.jp']

" ========== rhubarb.vim end ===========

" lua << EOF
"     require'lspconfig'.julials.setup{}
" EOF
" 
" autocmd Filetype julia setlocal omnifunc=v:lua.vim.lsp.omnifunc
" 
" nnoremap <silent> gd    <cmd>lua vim.lsp.buf.definition()<CR>
" nnoremap <silent> gD    <cmd>lua vim.lsp.buf.declaration()()<CR>
" nnoremap <silent> g0    <cmd>lua vim.lsp.buf.document_symbol()<CR>
" nnoremap <silent> gr    <cmd>lua vim.lsp.buf.references()<CR>
" nnoremap <silent> gi    <cmd>lua vim.lsp.buf.implementation()<CR>
" nnoremap <silent> K     <cmd>lua vim.lsp.buf.hover()<CR>

" ========== julia end ==========
