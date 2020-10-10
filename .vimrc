" The structure of this vimrc is following
" 1. dein Scripts
" 2. setting for vim itself
" 3. setting for standalone plugins
" 4. setting for programming languages
"
"dein Scripts-----------------------------
if &compatible
    set nocompatible               " Be iMproved
endif

" Required:
set runtimepath+=$HOME/.cache/dein/repos/github.com/Shougo/dein.vim

" Required:
if dein#load_state('$HOME/.cache/dein')
    call dein#begin('$HOME/.cache/dein')

    " Let dein manage dein
    " Required:
    call dein#add('$HOME/.local/share/dein/repos/github.com/Shougo/dein.vim')

    " Add or remove your plugins here:
    call dein#add('dag/vim2hs')
    call dein#add('tyru/caw.vim.git')
    call dein#add('Shougo/deoplete.nvim')
    if !has('nvim')
        call dein#add('roxma/nvim-yarp')
        call dein#add('roxma/vim-hug-neovim-rpc')
    endif
    call dein#add('Shougo/neosnippet')
    call dein#add('Shougo/neosnippet-snippets')
    call dein#add("vim-scripts/vim-auto-save")
    call dein#add('Shougo/unite.vim')
    call dein#add('Shougo/neomru.vim')
    call dein#add("Shougo/vimproc", {'build' : 'make'})
    call dein#add('tell-k/vim-autopep8')
    call dein#add('Shougo/vimfiler')
    call dein#add('def-lkb/ocp-indent-vim')
    call dein#add( 'akawashiro/the-ocamlspot.vim')
    call dein#add('kana/vim-filetype-haskell')
    call dein#add('eagletmt/ghcmod-vim')
    call dein#add('ujihisa/neco-ghc')
    call dein#add('zchee/deoplete-jedi')
    call dein#add('davidhalter/jedi-vim', {"autoload": { "filetypes": [ "python", "python3", "djangohtml"] }})
    call dein#add('vim-jp/cpp-vim')
    call dein#add('jvoorhis/coq.vim')
    call dein#add('vim-syntastic/syntastic')
    call dein#add('lervag/vimtex')
    call dein#add('vim-scripts/vim-auto-save')
    call dein#add('ujihisa/neco-look')
    call dein#add('tpope/vim-fugitive')
    call dein#add('scrooloose/nerdtree')
    call dein#add('junegunn/vim-easy-align')
    call dein#add('rust-lang/rust.vim')
    call dein#add('vim-airline/vim-airline')
    call dein#add('vim-airline/vim-airline-themes')
    call dein#add('haya14busa/incsearch.vim')
    call dein#add('fatih/vim-go')
    call dein#add('zchee/deoplete-go', {'build': 'make'})
    call dein#add('zchee/deoplete-clang')

    " Temporaly Commentouted Plugins
    " call dein#add('akawashiro/CoqIDE')
    " call dein#add('nbouscal/vim-stylish-haskell')

    " Suspicious Plugin neoyank
    " When I enable this plugins and select a region with V key, Vim: Caught deadly signal SEGV.
    " call dein#add('Shougo/neoyank.vim')



    " Required:
    call dein#end()
    call dein#save_state()
endif
" If you want to install not installed plugins on startup.
if dein#check_install()
    call dein#install()
endif
"End dein Scripts-------------------------


" NVim
let g:python3_host_prog = substitute(system('which python3'),"\n","","")

" Vim
filetype plugin indent on
syntax enable
set number
set shiftwidth=4
set tabstop=4
set expandtab
set cindent
syntax on
colorscheme desert

set nocompatible
set whichwrap=b,s,h,l,<,>,[,],~
set backspace=indent,eol,start

set clipboard=unnamedplus,autoselect
set hlsearch
nmap <Esc><Esc> :nohlsearch<CR><Esc>
inoremap jj <Esc>

set guifont =DejaVu\ Sans\ Mono\ 10 
" set guifontwide =VL\ ゴシック\ 10
set nofoldenable    " disable folding

" 行を強調表示
set cursorline
" 列を強調表示
" set cursorcolumn

" buffer
map <silent> tn :bnext<CR>
map <silent> tp :bprevious<CR>
map <silent> td :bdelete<CR>

" incsearch
map /  <Plug>(incsearch-forward)
map ?  <Plug>(incsearch-backward)
map g/ <Plug>(incsearch-stay)

" nerdtree
nnoremap <silent><C-e> :NERDTreeToggle<CR>

" vim-easy-align
" Start interactive EasyAlign in visual mode (e.g. vipga)
xmap ga <Plug>(EasyAlign)

" Start interactive EasyAlign for a motion/text object (e.g. gaip)
nmap ga <Plug>(EasyAlign)

" vim-auto-save
let g:auto_save = 1
let g:auto_save_in_insert_mode = 0

" syntastic
let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0

hi SyntasticErrorSign ctermfg=160
hi SyntasticWarningSign ctermfg=220

" vim-syntastic/syntastic
set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*

" caw comment out 
nmap <C-c> <Plug>(caw:hatpos:toggle)
vmap <C-c> <Plug>(caw:hatpos:toggle)

" vim-easy-align
" Start interactive EasyAlign in visual mode (e.g. vipga)
xmap ga <Plug>(EasyAlign)
" " Start interactive EasyAlign for a motion/text object (e.g. gaip)
nmap ga <Plug>(EasyAlign)

" neco-look english words complement
if !exists('g:neocomplete#text_mode_filetypes')
    let g:neocomplete#text_mode_filetypes = {}
endif
let g:neocomplete#text_mode_filetypes = {
            \ 'rst': 1,
            \ 'markdown': 1,
            \ 'gitrebase': 1,
            \ 'gitcommit': 1,
            \ 'vcs-commit': 1,
            \ 'hybrid': 1,
            \ 'text': 1,
            \ 'help': 1,
            \ 'tex': 1,
            \ }

" vim-fugitive
command -nargs=1 Gac Gwrite | Gcommit -m <args>
command -nargs=1 Gpo Git push origin <args>

" start airline --------------------------------------------------------

" Powerline系フォントを利用する
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

" end airline --------------------------------------------------------


" start deomplete --------------------------------------------------------
" Do not reverse the order of deomplete and neosnippet.
" They have confilict function and I want to prefer neosnippet function.
" Use deoplete.
let g:deoplete#enable_at_startup = 1

inoremap <expr><tab> pumvisible() ? "\<C-n>" :
            \ neosnippet#expandable_or_jumpable() ?
            \    "\<Plug>(neosnippet_expand_or_jump)" : "\<tab>"
" end   deomplete --------------------------------------------------------

" start neosnippet --------------------------------------------------------
" Plugin key-mappings.
imap <C-k>     <Plug>(neosnippet_expand_or_jump)
smap <C-k>     <Plug>(neosnippet_expand_or_jump)
xmap <C-k>     <Plug>(neosnippet_expand_target)

" SuperTab like snippets behavior.
"imap <expr><TAB>
" \ pumvisible() ? "\<C-n>" :
" \ neosnippet#expandable_or_jumpable() ?
" \    "\<Plug>(neosnippet_expand_or_jump)" : "\<TAB>"
smap <expr><TAB> neosnippet#expandable_or_jumpable() ?
            \ "\<Plug>(neosnippet_expand_or_jump)" : "\<TAB>"

" For conceal markers.
if has('conceal')
    set conceallevel=2 concealcursor=niv
endif
" end   neosnippet --------------------------------------------------------

" start unite --------------------------------------------------------
let g:unite_enable_start_insert=1
let g:unite_source_history_yank_enable =1
let g:unite_source_file_mru_limit = 200
nnoremap <silent> ,uy :<C-u>Unite history/yank<CR>
nnoremap <silent> ,ub :<C-u>Unite buffer<CR>
nnoremap <silent> ,uf :<C-u>UniteWithBufferDir -buffer-name=files file -default-action=tabopen<CR>
nnoremap <silent> ,ur :<C-u>Unite -buffer-name=register register<CR>
nnoremap <silent> ,uu :<C-u>Unite file_mru buffer file -default-action=tabopen<CR>

" grep
nnoremap <silent> ,g  :<C-u>Unite -default-action=tabopen grep:. -buffer-name=search-buffer<CR>
nnoremap <silent> ,cg :<C-u>Unite -default-action=tabopen grep:. -buffer-name=search-buffer<CR><C-R><C-W>
if executable('ag')
    let g:unite_source_grep_command = 'ag'
    let g:unite_source_grep_default_opts = '--nogroup --nocolor --column'
    let g:unite_source_grep_recursive_opt = ''
endif
" end   unite --------------------------------------------------------

" Start Egison ==============================================================
au BufRead,BufNewFile *.egi set filetype=egi
" End Egison ================================================================

" Start C++ ==============================================================

" syntastic for C++
let g:syntastic_cpp_compiler="g++"
let g:syntastic_cpp_compiler_options=" -std=c++0x"

" deoplete-clang
let g:deoplete#sources#clang#libclang_path = '/usr/lib/llvm-3.8/lib/libclang-3.8.so.1'
let g:deoplete#sources#clang#clang_header = '/usr/include/clang'
" End   C++ ==============================================================

" Start Haskell ==========================================================

" syntactic Setting for Haskell
let g:syntastic_haskell_checkers = ['ghc-mod']

"vim2hs
let g:haskell_conceal = 0

" End   Haskell ==========================================================

" Start Coq ==========================================================

" Maps Coquille commands to <F2> (Undo), <F3> (Next), <F4> (ToCursor)
" au FileType coq call coquille#FNMapping()
autocmd FileType coq highlight SentToCoq ctermbg=0 guibg=#000080
let g:CoqIDEDefaultMap = 1
" au BufRead,BufNewFile *.v   set filetype=coq

" End   Coq ==========================================================

" Start Python ==========================================================
let g:pymode_python = 'python3'

" syntastic checker python
let g:syntastic_python_checkers = ['flake8']

" End   Python ==========================================================

" Start Ocaml ==========================================================

let g:opamshare = substitute(system('opam config var share'),'\n$','','''')
execute "set rtp+=" . g:opamshare . "/merlin/vim"

let g:syntastic_ocaml_checkers = ['merlin']

if !exists('g:neocomplete#force_omni_input_patterns')
    let g:neocomplete#force_omni_input_patterns = {}
endif
let g:neocomplete#force_omni_input_patterns.ocaml = '[^.*\t]\.\w*\|\h\w*|#'

set rtp^="$HOME/.opam/system/share/ocp-indent/vim"
" execute 'set rtp^=' . g:opamshare . '/ocp-indent/vim'

let g:the_ocamlspot_disable_auto_type = 1

" End   Ocaml ==========================================================

" Start Latex ==========================================================
" tex mode
autocmd BufNewFile,BufRead *.otex setfiletype tex
let g:tex_conceal = ''
" End   Latex ==========================================================
"
