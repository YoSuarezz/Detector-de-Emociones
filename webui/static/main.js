// webui/static/main.js

const form = document.getElementById('upload-form');
const stdoutEl = document.getElementById('stdout');
const stderrEl = document.getElementById('stderr');
const outimg = document.getElementById('outimg');
const modalImg = document.getElementById('modalImg');
const spinner = document.getElementById('spinner');
const badges = document.getElementById('badges');
const genderBadge = document.getElementById('gender-badge');
const historyEl = document.getElementById('history');
const clearHistoryBtn = document.getElementById('clear-history');

let emoChart = null;
let genChart = null;

function showSpinner(v=true){
  // Usamos inline style para controlar display y flex
  if(v){
    spinner.style.display = 'flex';
  } else {
    spinner.style.display = 'none';
  }
}

function initCharts(){
  const emoCtx = document.getElementById('emoChart');
  emoChart = new Chart(emoCtx, {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'Prob', data: [] }] },
    options: { responsive:true, animation:false, scales:{ y:{ beginAtZero:true, max:1 } } }
  });

  const genCtx = document.getElementById('genChart');
  genChart = new Chart(genCtx, {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'Prob', data: [] }] },
    options: { responsive:true, animation:false, scales:{ y:{ beginAtZero:true, max:1 } } }
  });
}

function renderCharts(emo_probs, gen_probs){
  const emo_labels = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised'];
  emoChart.data.labels = emo_labels;
  emoChart.data.datasets[0].data = emo_probs;
  emoChart.update();

  const gen_labels = ['female','male'];
  genChart.data.labels = gen_labels;
  genChart.data.datasets[0].data = gen_probs;
  genChart.update();
}

function pushHistory(item){
  let h = JSON.parse(localStorage.getItem('infer_history') || '[]');
  h.unshift(item);
  if(h.length>6) h = h.slice(0,6);
  localStorage.setItem('infer_history', JSON.stringify(h));
  renderHistory();
}

function renderHistory(){
  const h = JSON.parse(localStorage.getItem('infer_history') || '[]');
  historyEl.innerHTML = '';
  for(const it of h){
    const li = document.createElement('li');
    li.className = 'list-group-item';
    li.textContent = `${it.filename} — ${it.summary} — ${new Date(it.t).toLocaleTimeString()}`;
    historyEl.appendChild(li);
  }
}

clearHistoryBtn.addEventListener('click', ()=>{ localStorage.removeItem('infer_history'); renderHistory(); });

outimg.addEventListener('click', ()=>{
  if(outimg.src){ modalImg.src = outimg.src; var m = new bootstrap.Modal(document.getElementById('imgModal')); m.show(); }
});

form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  // Primero validamos archivo, si no hay archivo no mostramos spinner
  const fileInput = document.getElementById('wavfile');
  if (fileInput.files.length === 0){
    stderrEl.textContent = 'Selecciona un archivo WAV antes de enviar.';
    return;
  }

  // Ahora sí mostramos spinner y limpiamos áreas
  showSpinner(true);
  stdoutEl.textContent = '';
  stderrEl.textContent = '';
  badges.innerHTML = '';
  genderBadge.innerHTML = '';

  const ckptInput = document.getElementById('ckpt');
  const fd = new FormData();
  fd.append('wavfile', fileInput.files[0]);
  fd.append('ckpt', ckptInput.value || 'model_final.pth');

  try{
    const res = await fetch('/upload', { method: 'POST', body: fd });
    // si ocurriese un error de red, esto lanzaría excepción
    const j = await res.json();
    showSpinner(false);
    if(!j.ok){ stderrEl.textContent = j.error || j.stderr || 'Error'; return; }

    stdoutEl.textContent = j.stdout || '';
    stderrEl.textContent = j.stderr || '';

    if(j.top_emotions){
      j.top_emotions.forEach(([lab, p]) =>{
        const s = document.createElement('span');
        s.className = 'badge rounded-pill bg-primary badge-emo';
        s.textContent = `${lab} ${(p*100).toFixed(1)}%`;
        badges.appendChild(s);
      });
    }
    if(j.top_gender){
      const [glabel, gp] = j.top_gender;
      const s = document.createElement('span');
      s.className = 'badge rounded-pill bg-success';
      s.textContent = `${glabel} ${(gp*100).toFixed(1)}%`;
      genderBadge.appendChild(s);
    }

    if(j.emo_probs && j.gen_probs){
      renderCharts(j.emo_probs, j.gen_probs);
    }

    if(j.image){
      outimg.src = j.image + '?_ts=' + Date.now();
      outimg.style.display = 'block';
    } else {
      outimg.style.display = 'none';
    }

    pushHistory({ filename: fileInput.files[0].name, summary: j.stdout || 'ok', t: Date.now() });

  }catch(e){
    showSpinner(false);
    stderrEl.textContent = e.toString();
  }
});

// init
initCharts(); renderHistory();
