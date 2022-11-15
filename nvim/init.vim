if !&compatible
  set nocompatible
endif

" reset augroup
augroup MyAutoCmd
  autocmd!
augroup END

let data_dir = has('nvim') ? stdpath('data') . '/site' : '~/.vim'
if empty(glob(data_dir . '/autoload/plug.vim'))
  silent execute '!curl -fLo '.data_dir.'/autoload/plug.vim --create-dirs  https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
  autocmd VimEnter * PlugInstall --sync | source $MYVIMRC
endif

" ========== vim-plug start ==========

call plug#begin()

Plug 'easymotion/vim-easymotion'
Plug 'tpope/vim-rhubarb'
Plug 'mechatroner/rainbow_csv'
Plug 'chrisbra/Colorizer'
Plug 'junegunn/fzf', { 'dir': '~/.fzf', 'do': './install --all' }
Plug 'junegunn/fzf.vim'
Plug 'pboettch/vim-cmake-syntax'
Plug 'scrooloose/nerdtree'
Plug 'jistr/vim-nerdtree-tabs'
Plug 'preservim/tagbar'
Plug 'airblade/vim-gitgutter'
Plug 'cespare/vim-toml'
Plug 'tyru/caw.vim'
Plug 'tpope/vim-fugitive'
Plug 'neoclide/coc.nvim', {'branch': 'release'}
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'

call plug#end()

" ========== vim-plug settings end ==========

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

colorscheme desert

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

" ========== julia start ==========

autocmd BufNewFile,BufRead *.jl setfiletype julia

" ========== julia end ==========

" ========== rhubarb.vim start ===========

let g:github_enterprise_urls = ['https://github.pfidev.jp']

" ========== rhubarb.vim end ===========

" ========== clang_format start ===========

let g:clang_format#auto_format = 0

" ========== clang_format end ===========

" ========== fzf start ===========

let $FZF_DEFAULT_OPTS="--layout=reverse --border"
let g:fzf_colors = {
    \ 'fg':      ['fg', 'Normal'],   
    \ 'bg':      ['bg', 'Normal'],   
    \ 'hl':      ['fg', 'Comment'],   
    \ 'fg+':     ['fg', 'CursorLine', 'CursorColumn', 'Normal'],   
    \ 'bg+':     ['bg', 'CursorLine', 'CursorColumn'],   
    \ 'hl+':     ['fg', 'Statement'],   
    \ 'info':    ['fg', 'PreProc'],   
    \ 'border':  ['fg', 'Keyword'],   
    \ 'prompt':  ['fg', 'Conditional'],   
    \ 'pointer': ['fg', 'Exception'],   
    \ 'marker':  ['fg', 'Keyword'],   
    \ 'spinner': ['fg', 'Label'],   
    \ 'header':  ['fg', 'Comment'] }

command!      -bang -nargs=? -complete=dir TmuxFiles       call fzf#vim#files(<q-args>, fzf#vim#with_preview({'tmux': '-p80%'}), <bang>0)',
command!      -bang -nargs=? TmuxGitFiles                  call fzf#vim#gitfiles(<q-args>, fzf#vim#with_preview(<q-args> == "?" ? { "placeholder": "", 'tmux': '-p80%' } : {'tmux': '-p80%'}), <bang>0)',
command!      -bang -nargs=? TmuxGFiles                    call fzf#vim#gitfiles(<q-args>, fzf#vim#with_preview(<q-args> == "?" ? { "placeholder": "", 'tmux': '-p80%' } : {'tmux': '-p80%'}), <bang>0)',
command! -bar -bang -nargs=? -complete=buffer TmuxBuffers  call fzf#vim#buffers(<q-args>, fzf#vim#with_preview({ "placeholder": "{1}", 'tmux': '-p80%' }), <bang>0)',
command!      -bang -nargs=* TmuxAg                        call fzf#vim#ag(<q-args>, fzf#vim#with_preview({'tmux': '-p80%'}), <bang>0)',
command! -bang -nargs=* TmuxGCGrep call fzf#vim#grep('git grep --line-number -- '.shellescape(expand(<q-args>)), 0, fzf#vim#with_preview({'options': ['--query', expand('<cword>')], 'dir': systemlist('git rev-parse --show-toplevel')[0], 'tmux': '-p80%'}), <bang>0)
command! -bang -nargs=* TmuxGGrep
  \ call fzf#vim#grep(
  \   'git grep --line-number -- '.shellescape(<q-args>), 0,
  \   fzf#vim#with_preview({'dir': systemlist('git rev-parse --show-toplevel')[0], 'options': '--delimiter : --nth 3..', 'tmux': '-p80%'}), <bang>0)

noremap <Leader>ff :TmuxFiles<CR>
noremap <Leader>fgf :TmuxGFiles<CR>
noremap <Leader>fgg :TmuxGGrep<CR>
noremap <Leader>fgc :TmuxGCGrep<CR>
noremap <Leader>fa :TmuxAg<CR>
noremap <Leader>fb :TmuxBuffers<CR>

" ========== fzf end ===========

" ========== nerdtree start ===========

let NERDTreeShowHidden = 1
noremap <Leader>nt :NERDTreeTabsToggle<CR>
noremap <Leader>nf :NERDTreeFind<CR>

" ========== nerdtree end ===========

nmap <C-c> <Plug>(caw:hatpos:toggle)
vmap <C-c> <Plug>(caw:hatpos:toggle)

" ========== airline start ==========

" https://github.com/neoclide/coc.nvim/issues/1827#issuecomment-621204299
let g:airline#extensions#hunks#enabled = 0

" Use powerline-style font
set laststatus=2
let g:airline_powerline_fonts = 1
let g:airline#extensions#tabline#enabled = 1
let g:airline#extensions#tabline#buffer_idx_mode = 1
let g:airline#extensions#whitespace#mixed_indent_algo = 1
let g:airline_theme = 'tomorrow'
if !exists('g:airline_symbols')
  let g:airline_symbols = {}
endif

" unicode symbols
let g:airline_left_sep = '»'
let g:airline_left_sep = '▶'
let g:airline_right_sep = '«'
let g:airline_right_sep = '◀'
let g:airline_symbols.crypt = '🔒'
let g:airline_symbols.linenr = '☰'
let g:airline_symbols.linenr = '␊'
let g:airline_symbols.linenr = '␤'
let g:airline_symbols.linenr = '¶'
let g:airline_symbols.maxlinenr = ''
let g:airline_symbols.maxlinenr = '㏑'
let g:airline_symbols.branch = '⎇'
let g:airline_symbols.paste = 'ρ'
let g:airline_symbols.paste = 'Þ'
let g:airline_symbols.paste = '∥'
let g:airline_symbols.spell = 'Ꞩ'
let g:airline_symbols.notexists = '∄'
let g:airline_symbols.whitespace = 'Ξ'

" powerline symbols
let g:airline_left_sep = ''
let g:airline_left_alt_sep = ''
let g:airline_right_sep = ''
let g:airline_right_alt_sep = ''
let g:airline_symbols.branch = ''
let g:airline_symbols.readonly = ''
let g:airline_symbols.linenr = '☰'
let g:airline_symbols.maxlinenr = ''

" ========== airline end ==========

" ========== start coc ==========

" May need for vim (not neovim) since coc.nvim calculate byte offset by count
" utf-8 byte sequence.
set encoding=utf-8
" Some servers have issues with backup files, see #649.
set nobackup
set nowritebackup

" Having longer updatetime (default is 4000 ms = 4 s) leads to noticeable
" delays and poor user experience.
set updatetime=4000

" Always show the signcolumn, otherwise it would shift the text each time
" diagnostics appear/become resolved.
set signcolumn=yes

" default install
" Installing coc-rust-analyzer causes some conflict with coc-clangd
let g:coc_global_extensions = [
    \'coc-clangd',
    \'coc-pyright',
    \]

" Color of Error and Warning
highlight CocErrorSign ctermfg=15 ctermbg=196
highlight CocWarningSign ctermfg=0 ctermbg=172

" Jump to the definition with \-d
nmap <silent> <Leader>d <Plug>(coc-definition)
" Run format with \-fmt
nmap <silent> <Leader>fmt <Plug>(coc-format)
" Coc-quick-fix
nnoremap <silent> <Leader>fx <Plug>(coc-fix-current)

" Use K to show documentation in preview window.
nnoremap <silent> K :call ShowDocumentation()<CR>

function! ShowDocumentation()
  if CocAction('hasProvider', 'hover')
    call CocActionAsync('doHover')
  else
    call feedkeys('K', 'in')
  endif
endfunction

" Add (Neo)Vim's native statusline support.
" NOTE: Please see `:h coc-status` for integrations with external plugins that
" provide custom statusline: lightline.vim, vim-airline.
set statusline^=%{coc#status()}%{get(b:,'coc_current_function','')}

" ========== end coc ==========
