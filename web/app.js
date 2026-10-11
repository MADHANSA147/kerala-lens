const $ = id => document.getElementById(id);
let selected, previewURL, result;
function choose(file) {
  if (!file) return;
  selected = file; result = null;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file);
  $('preview').src = previewURL;
  $('preview').hidden = false; $('placeholder').hidden = true;
  $('filename').textContent = file.name;
  $('answer').hidden = true; $('empty').hidden = false; $('error').hidden = true;
  $('analyse').disabled = false;
}
$('file').addEventListener('change', e => choose(e.target.files[0]));
['dragover','dragleave','drop'].forEach(type => $('drop').addEventListener(type, e => {
  e.preventDefault(); $('drop').classList.toggle('drag', type === 'dragover');
  if (type === 'drop' && !$('file').disabled) choose(e.dataTransfer.files[0]);
}));
$('analyse').addEventListener('click', async () => {
  if (!selected) return;
  $('analyse').disabled = true; $('file').disabled = true;
  $('empty').hidden = true; $('answer').hidden = true; $('error').hidden = true; $('loading').hidden = false;
  try {
    const form = new FormData(); form.append('image', selected);
    const response = await fetch('/api/predict', {method:'POST', body:form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not analyse this photo.');
    result = data; $('label').textContent = data.label; $('reason').textContent = data.reason;
    $('coordinates').replaceChildren();
    data.candidates.forEach((c, i) => {
      const row = document.createElement('tr');
      [i+1,c.latitude.toFixed(4),c.longitude.toFixed(4),c.in_kerala?'Kerala':'Outside Kerala'].forEach(value => {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      });
      $('coordinates').append(row);
    });
    $('answer').hidden = false;
  } catch(error) { $('error').textContent = error.message; $('error').hidden = false; }
  finally { $('loading').hidden = true; $('analyse').disabled = false; $('file').disabled = false; }
});
$('save').addEventListener('click', () => {
  if (!result) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(result,null,2)], {type:'application/json'}));
  const a = document.createElement('a'); a.href=url; a.download='kerala-lens-result.json'; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
});
fetch('/api/report').then(r=> {if(!r.ok) throw new Error(); return r.json();}).then(r=> {
  $('accuracy').textContent = (r.held_out_test_forced_binary.accuracy_on_accepted*100).toFixed(1)+'%';
  $('uncertain').textContent = r.held_out_test.uncertain+' / '+r.held_out_test.images;
  $('distance').textContent = Math.round(r.held_out_geoclip_coordinate_error.median_km)+' km';
}).catch(()=> { $('accuracy').textContent = 'Unavailable'; });
