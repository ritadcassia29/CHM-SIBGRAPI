const fs = require('fs');
const input = fs.readFileSync('SIBGRAPI_REV/GUIA_EXPLICATIVO_PROJETO.md','utf8').replace(/^\uFEFF/,'');
function esc(s){ return s.replace(/\\/g,'\\textbackslash{}').replace(/([#$%&_{}])/g,'\\$1').replace(/~/g,'\\textasciitilde{}').replace(/\^/g,'\\textasciicircum{}'); }
function inline(s){
  let out='', i=0;
  const re=/\[([^\]]+)\]\(([^)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|\*([^*]+)\*/g; let m;
  while((m=re.exec(s))){ out+=esc(s.slice(i,m.index));
    if(m[1]!==undefined) out+='\\href{'+m[2].replace(/ /g,'%20')+'}{'+inline(m[1])+'}';
    else if(m[3]!==undefined) out+='\\url{'+m[3].replace(/%/g,'%25')+'}';
    else if(m[4]!==undefined) out+='\\textbf{'+inline(m[4])+'}';
    else out+='\\emph{'+inline(m[5])+'}'; i=re.lastIndex;
  }
  return out+esc(s.slice(i));
}
const lines=input.split(/\r?\n/); let body=[], para=[], list=false, ordered=false, quote=false, table=false, cols=0;
function closePara(){if(para.length){body.push(inline(para.join(' '))+'\n');para=[];}}
function endList(){if(list){body.push(ordered?'\\end{enumerate}':'\\end{itemize}');list=false;ordered=false;}}
function endQuote(){if(quote){body.push('\\end{quote}');quote=false;}}
function endTable(){if(table){body.push('\\end{tabularx}\\end{center}');table=false;}}
for(let i=0;i<lines.length;i++){
 let s=lines[i].trim();
 if(!s){closePara();endList();endQuote();endTable();continue;}
 if(/^\|.*\|$/.test(s)){
   closePara();endList();endQuote();
   const cells=s.slice(1,-1).split('|').map(x=>x.trim());
   if(cells.every(x=>/^:?-{3,}:?$/.test(x))) continue;
   if(!table){cols=cells.length;body.push('\\begin{center}\\small\\begin{tabularx}{\\textwidth}{|'+Array(cols).fill('X').join('|')+'|}\\hline');table=true;}
   body.push(cells.map(inline).join(' & ')+' \\\\ \\hline'); continue;
 }
 endTable();
 let m;
 if((m=s.match(/^(#{1,3})\s+(.*)$/))){closePara();endList();endQuote();let level=m[1].length;body.push(level===1?'\\section{'+inline(m[2])+'}':level===2?'\\subsection{'+inline(m[2])+'}':'\\subsubsection{'+inline(m[2])+'}');continue;}
 if(/^---+$/.test(s)){closePara();endList();endQuote();body.push('\\medskip\\hrule\\medskip');continue;}
 if(s.startsWith('>')){closePara();endList();if(!quote){body.push('\\begin{quote}\\color{cinza}');quote=true;}body.push(inline(s.replace(/^>\s?/,''))+'\\par');continue;}
 if((m=s.match(/^[-*]\s+(.*)$/))){closePara();endQuote();if(list&&ordered)endList();if(!list){body.push('\\begin{itemize}');list=true;ordered=false;}body.push('\\item '+inline(m[1]));continue;}
 if((m=s.match(/^\d+\.\s+(.*)$/))){closePara();endQuote();if(list&&!ordered)endList();if(!list){body.push('\\begin{enumerate}');list=true;ordered=true;}body.push('\\item '+inline(m[1]));continue;}
 endQuote();para.push(s);
}
closePara();endList();endQuote();endTable();
const pre=`\\documentclass[11pt,a4paper]{article}
\\usepackage[a4paper,margin=2.25cm,headheight=15pt]{geometry}
\\usepackage{fontspec}
\\setmainfont{Arial}
\\usepackage{polyglossia}
\\setdefaultlanguage{brazil}
\\usepackage{xcolor}
\\definecolor{azul}{HTML}{174A6E}
\\definecolor{cinza}{HTML}{4B5563}
\\usepackage{titlesec}
\\titleformat{\\section}{\\Large\\bfseries\\color{azul}}{\\thesection}{0.6em}{}
\\titleformat{\\subsection}{\\large\\bfseries\\color{azul}}{\\thesubsection}{0.6em}{}
\\titleformat{\\subsubsection}{\\normalsize\\bfseries\\color{azul}}{\\thesubsubsection}{0.6em}{}
\\setcounter{secnumdepth}{0}
\\usepackage{enumitem}
\\setlist{itemsep=2pt,topsep=4pt,leftmargin=1.6em}
\\usepackage{microtype}
\\usepackage{hyperref}
\\usepackage{tabularx}
\\hypersetup{colorlinks=true,linkcolor=azul,urlcolor=azul,pdftitle={Guia de leitura do projeto CHM-SIBGRAPI}}
\\usepackage{fancyhdr}
\\pagestyle{fancy}\\fancyhf{}
\\fancyhead[L]{\\small\\color{cinza}CHM-SIBGRAPI | Guia de leitura}
\\fancyhead[R]{\\small\\color{cinza}Redes neurais e sensoriamento remoto}
\\fancyfoot[C]{\\thepage}
\\setlength{\\parskip}{5pt}\\setlength{\\parindent}{0pt}
\\begin{document}
`;
fs.writeFileSync('output/pdf/GUIA_EXPLICATIVO_PROJETO.tex',pre+body.join('\n')+'\n\\end{document}\n','utf8');

