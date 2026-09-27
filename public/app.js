const $ = id => document.getElementById(id);
const definitions = {
  quantize: { help: 'K-means thay mỗi điểm ảnh bằng màu đại diện. So sánh ảnh gốc với K = 2, 4, 8, 16.', params: ['max_side'] },
  segment: { help: 'So sánh phân cụm theo màu RGB và theo màu kết hợp vị trí. Các màu biểu diễn nhãn cụm, không phải tên vật thể.', params: ['k','weight','max_side'] },
  spectral: { help: 'Cùng một ảnh được cắt vuông ở giữa và thu nhỏ 64 × 64. Spectral sử dụng đồ thị 12 láng giềng.', params: ['k','weight'], defaults: {k:3,weight:0.5} },
  faces: { help: 'Haar cascade có sẵn trong OpenCV. Khung xanh biểu diễn vùng mặt, khung vàng biểu diễn mắt. Chọn mặt chính diện, đủ sáng.', params:['scale','neighbors','min_size','max_side'] },
  'face-parameters': { help:'So sánh 4 tổ hợp scaleFactor 1,05/1,20 và minNeighbors 3/8. Kết quả có thể giống nhau trên một số ảnh.', params:['min_size','max_side'] },
  hog: { help:'Biểu diễn gradient với 9 khoảng hướng, khối 2 × 2 ô và chuẩn L2-Hys. HOG mô tả đường nét, chưa nhận dạng đối tượng.', params:['cell','max_side'] },
  'hog-svm': { help:'SVM đã huấn luyện bằng ảnh mẫu. Quét ảnh với đặc trưng HOG rồi loại hộp chồng lấn bằng NMS. Điểm SVM không phải xác suất; có thể bỏ sót hoặc báo nhầm.', params:['min_size','threshold','iou','max_side'], defaults:{min_size:60} }
};
const fields = {
  k:{label:'Số cụm K',value:4,min:2,max:16,step:1}, weight:{label:'Trọng số vị trí',value:0.6,min:0,max:2,step:0.1},
  max_side:{label:'Cạnh xử lý tối đa',value:512,values:[256,384,512]}, scale:{label:'scaleFactor',value:1.1,min:1.01,max:2,step:0.01},
  neighbors:{label:'minNeighbors',value:5,min:0,max:30,step:1}, min_size:{label:'Mặt tối thiểu (px)',value:30,min:10,max:500,step:1},
  cell:{label:'Kích thước ô HOG',value:16,values:[4,8,16,32]}, threshold:{label:'Ngưỡng điểm SVM',value:0.6,min:0,max:3,step:0.05},
  iou:{label:'Ngưỡng IoU NMS',value:0.25,min:0.05,max:1,step:0.05}
};
let source = null, result = null, busy = false, preparing = false, selection = 0;
function status(message, kind='') { $('status').textContent=message; $('status').className='status '+kind; }
function renderParameters(){
  const method=$('method').value, config=definitions[method];
  $('method-help').textContent=config.help; $('params').replaceChildren();
  for(const key of config.params){
    const spec=fields[key], wrapper=document.createElement('div'); wrapper.className='param';
    const label=document.createElement('label'); label.htmlFor='param-'+key; label.textContent=spec.label;
    const input=document.createElement(spec.values?'select':'input'); input.id=label.htmlFor; input.name=key;
    if(spec.values){for(const value of spec.values){const option=document.createElement('option'); option.value=value; option.textContent=key==='cell'?`${value} × ${value}`:`${value} px`; input.append(option);}}
    else {input.type='number'; input.min=(key==='min_size'&&method==='hog-svm')?40:spec.min; input.max=spec.max; input.step=spec.step; input.required=true;}
    input.value=config.defaults?.[key]??spec.value; wrapper.append(label,input); $('params').append(wrapper);
  }
}
function availability(){ $('run').disabled=!source||busy||preparing; $('sample').disabled=busy||preparing; $('file').disabled=busy; $('method').disabled=busy; }
function resetResult(){ result=null; $('result-section').hidden=true; }
async function readFile(file){
  if(busy)return; const token=++selection; preparing=true; source=null; resetResult(); availability();
  $('original').hidden=true; $('placeholder').hidden=false;
  try{
    if(file.size>25*1024*1024)throw new Error('Chọn ảnh nhỏ hơn 25 MB.');
    if(!['image/png','image/jpeg','image/webp'].includes(file.type))throw new Error('Chọn ảnh JPG, PNG hoặc WebP.');
    status('Đang chuẩn bị ảnh…','busy');
    const bitmap=await createImageBitmap(file,{imageOrientation:'from-image'});
    const originalSize=[bitmap.width,bitmap.height];
    const ratio=Math.min(1,512/Math.max(bitmap.width,bitmap.height));
    const canvas=document.createElement('canvas'); canvas.width=Math.max(1,Math.round(bitmap.width*ratio)); canvas.height=Math.max(1,Math.round(bitmap.height*ratio));
    const context=canvas.getContext('2d'); context.fillStyle='white'; context.fillRect(0,0,canvas.width,canvas.height); context.drawImage(bitmap,0,0,canvas.width,canvas.height); bitmap.close();
    const uri=canvas.toDataURL('image/png');
    if(token!==selection)return;
    source={uri,name:file.name,originalSize,sentSize:[canvas.width,canvas.height]};
    $('original').src=uri; $('original').hidden=false; $('placeholder').hidden=true;
    $('file-info').textContent=`${file.name} · gốc ${originalSize.join(' × ')} px`;
    $('dimensions').textContent=`Ảnh gửi xử lý: ${source.sentSize.join(' × ')} px`;
    status('Ảnh đã sẵn sàng. Chọn chức năng và bấm Xử lý ảnh.');
  }catch(error){if(token===selection){$('file-info').textContent='Chưa chọn được ảnh.';status(error.message||'Không mở được ảnh.','error');}}
  finally{if(token===selection){preparing=false;availability();}}
}
$('file').addEventListener('change',()=>{if($('file').files[0])readFile($('file').files[0]);});
$('method').addEventListener('change',()=>{renderParameters();resetResult();status(source?'Chọn tham số rồi bấm Xử lý ảnh.':'Mở ảnh để bắt đầu.');});
$('sample').addEventListener('click',async()=>{try{const response=await fetch('/sample.jpg');if(!response.ok)throw new Error('Không tải được ảnh mẫu.');await readFile(new File([await response.blob()],'anh_mau_scikit_image.jpg',{type:'image/jpeg'}));}catch(error){status(error.message,'error');}});
for(const event of ['dragenter','dragover'])$('dropzone').addEventListener(event,e=>{e.preventDefault();if(!busy)$('dropzone').classList.add('drag');});
for(const event of ['dragleave','drop'])$('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.remove('drag');if(event==='drop'&&e.dataTransfer.files[0])readFile(e.dataTransfer.files[0]);});
$('form').addEventListener('submit',async event=>{
  event.preventDefault();if(!source||busy||preparing)return;
  const method=$('method').value, options={};
  for(const key of definitions[method].params)options[key]=Number($('param-'+key).value);
  const submitted={...source}; busy=true;resetResult();availability();
  for(const input of $('params').querySelectorAll('input,select'))input.disabled=true;
  $('run').textContent='Đang xử lý…';
  const started=Date.now(),controller=new AbortController();
  const timer=setInterval(()=>status(`Đang xử lý ảnh · ${Math.floor((Date.now()-started)/1000)} giây. Lần đầu có thể lâu hơn do máy chủ khởi động.`,'busy'),1000);
  const timeout=setTimeout(()=>controller.abort(),70000);
  status('Đang gửi ảnh và chạy thuật toán…','busy');
  try{
    const response=await fetch('/api/process',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({method,options,image:submitted.uri.split(',')[1]}),signal:controller.signal});
    let data;try{data=await response.json();}catch{throw new Error(response.status===504?'Máy chủ xử lý quá lâu. Thử ảnh nhỏ hơn hoặc giảm phạm vi quét.':'Máy chủ không trả về kết quả hợp lệ. Thử lại sau.');}
    if(!response.ok)throw new Error(data.error||'Không xử lý được ảnh.');
    data.metadata.source_file=submitted.name; data.metadata.client_original_size=submitted.originalSize; data.metadata.client_prepared_size=submitted.sentSize;
    result=data; $('result').src=data.image; $('result-title').textContent=data.title; $('timing').textContent=`${data.seconds} giây`;
    $('description').textContent=data.description; $('notes').textContent=data.notes;
    $('metadata').textContent=JSON.stringify(data.metadata,null,2); $('panel-downloads').replaceChildren();
    data.panels.forEach((panel,index)=>{const button=document.createElement('button');button.type='button';button.textContent=`↓ ${panel.title}`;button.addEventListener('click',()=>download(panel.image,`${method}_${index+1}.png`));$('panel-downloads').append(button);});
    $('result-section').hidden=false;status('Xử lý hoàn tất. Bạn có thể xem và tải kết quả.');
  }catch(error){status(error.name==='AbortError'?'Đã hết thời gian chờ. Giảm kích thước ảnh hoặc thử lại sau.':error.message,'error');}
  finally{clearInterval(timer);clearTimeout(timeout);busy=false;availability();for(const input of $('params').querySelectorAll('input,select'))input.disabled=false;$('run').textContent='Xử lý ảnh →';}
});
function download(uri,filename){const a=document.createElement('a');a.href=uri;a.download=filename;document.body.append(a);a.click();a.remove();}
function downloadText(text,filename,type){const uri=URL.createObjectURL(new Blob([text],{type}));download(uri,filename);setTimeout(()=>URL.revokeObjectURL(uri),1000);}
$('download-image').addEventListener('click',()=>{if(result)download(result.image,`${result.method}_so_sanh.png`);});
$('download-json').addEventListener('click',()=>{if(result)downloadText(JSON.stringify(result.metadata,null,2),`${result.method}_thong_so.json`,'application/json');});
$('download-notes').addEventListener('click',()=>{if(result)downloadText('\ufeff'+result.description+'\n\n'+result.notes,`${result.method}_mo_ta.txt`,'text/plain;charset=utf-8');});
renderParameters();
