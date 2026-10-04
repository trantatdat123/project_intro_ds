import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

// Chạy bản sao trong .build/slides có junction node_modules của runtime.
const root = process.env.JOBLENS_ROOT || process.cwd();
const skill = process.env.JOBLENS_PRESENTATION_SKILL;
const python = process.env.JOBLENS_RUNTIME_PYTHON;
if (!skill || !python) throw new Error('Thiếu runtime/skill; xem scripts/README.md');
const build = path.join(root, '.build/slides');
const output = process.env.JOBLENS_SLIDES_OUTPUT || path.join(root, 'reports/slides/joblens_vietnam.pptx');
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const family = resolvePresentationFont({fontFamily:'Arial'});
const fontPolicy = { basis: 'design', families: [family] };
const deck = Presentation.create({slideSize:{width:1280,height:720}});
const C = {ink:'#153044', teal:'#176B87', green:'#359F87', orange:'#DE9B43', muted:'#4B6373', paper:'#FFFFFF'};
const json = async name => JSON.parse(await fs.readFile(path.join(root,'reports',name),'utf8'));
function parseCsv(text) {
  const rows=[]; let row=[], field='', quoted=false;
  for(let i=0;i<text.length;i++) {
    const ch=text[i];
    if(ch==='"') { if(quoted && text[i+1]==='"') {field+='"'; i++;} else quoted=!quoted; }
    else if(ch===',' && !quoted) {row.push(field);field='';}
    else if(ch==='\n' && !quoted) {row.push(field.replace(/\r$/,'')); rows.push(row);row=[];field='';}
    else field+=ch;
  }
  if(field || row.length) {row.push(field);rows.push(row);}
  const header=rows.shift().map(x=>x.replace(/^\uFEFF/,''));
  return rows.filter(r=>r.length===header.length).map(r=>Object.fromEntries(header.map((h,i)=>[h,r[i]])));
}
const csv = async name=>parseCsv(await fs.readFile(path.join(root,'reports',name),'utf8'));
const quality=await json('data_quality.json');
const cls=await json('classification_metrics.json');
const reg=await json('salary_metrics.json');
const clustering=await json('clustering_metrics.json');
const manifest=await json('run_manifest.json');
const config=manifest.config;
const provenance=JSON.parse(await fs.readFile(path.join(root,config.input.replace(/\.parquet$/,'.metadata.json')),'utf8'));
if(!clustering.semantic.kmeans) throw new Error('Chưa có kết quả semantic clustering');
const years=await csv('tables/year_distribution.csv');
const roles=await csv('tables/role_distribution.csv');
const skills=await csv('tables/skill_frequency.csv');
const salaries=await csv('tables/salary_by_role.csv');
const centrality=await csv('skill_centrality.csv');
const shap=await csv('salary_shap_importance.csv');
const source='https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions';
const number=n=>Number(n).toLocaleString('vi-VN');
const shortRole=x=>x.replace('Software Engineer','Software').replace('AI/ML Engineer','AI/ML');
const metric=n=>Number(n).toFixed(3);
const million=n=>(Number(n)/1e6).toFixed(2);
const bestCls=cls.models[cls.selected_model].test;
const bestReg=reg.models[reg.selected_model].test;
const charts=[], tables=[];

function text(s,value,x,y,w,h,size=26,bold=false,color=C.ink) {
  const shape=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=value;
  shape.text.style={typeface:family,fontSize:size,bold,color,autoFit:'none',wrap:true,insets:{left:0,right:0,top:0,bottom:0}};
  return shape;
}
function slide(title,notes='') {
  const s=deck.slides.add();s.background.fill=C.paper;
  text(s,title,72,42,1136,92,44,true);
  s.speakerNotes.textFrame.setText(notes);
  return s;
}
function note(s,value) { text(s,value,72,630,1136,64,23,false,C.muted); }
function table(s,values,y=180,widths=[560,270,270],h=310) {
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:72,top:y,width:1136,height:h,values,columnWidths:widths});
  t.borders.assign({fill:'#CFDAE0',width:1,style:'solid'});
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({
    textStyle:{typeface:family,fontSize:25,color:C.ink},margins:{left:14,right:14,top:10,bottom:10},anchor:'center'});
  for(let c=0;c<values[0].length;c++) {t.getCell(0,c).fill=C.teal;t.getCell(0,c).text.style={typeface:family,fontSize:25,bold:true,color:'#FFFFFF'};}
  tables.push(deck.slides.items.length);
  return t;
}
function bar(s,categories,series,{horizontal=false,y=162,h=432,format='#,##0',max,legend=false}={}) {
  const chart=s.charts.add('bar',{position:{left:72,top:y,width:1136,height:h},categories,
    series:series.map((v,i)=>({...v,values:v.values.map(n=>Number(Number(n).toFixed(6))),
                               valuesFormatCode:format,fill:[C.teal,C.green,C.orange][i]})),
    hasLegend:legend,legend:{position:'top',textStyle:{fontSize:24,fill:C.ink,typeface:family}},
    barOptions:{direction:horizontal?'bar':'column',grouping:'clustered',gapWidth:65},
    xAxis:{textStyle:{fontSize:24,fill:C.ink,typeface:family},majorGridlines:null},
    yAxis:{textStyle:{fontSize:24,fill:C.muted,typeface:family},numberFormatCode:format,
           ...(max===undefined?{}:{max}),majorGridlines:{fill:'#E5EBEF',width:1}},
    dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:23,fill:C.ink,typeface:family}},
    chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'});
  applyPresentationChartFont(chart,{fontFamily:family});
  charts.push(deck.slides.items.length);
  return chart;
}

// 1 — cover
{
 const s=deck.slides.add();s.background.fill=C.paper;
 text(s,'JobLens Vietnam',72,230,1136,108,68,true,C.teal);
 text(s,'Tin tuyển dụng IT, AI và Data Science',72,350,1136,80,34);
 text(s,'Dự án Introduction to Data Science',72,456,1136,62,26,false,C.muted);
 s.speakerNotes.textFrame.setText(`Nguồn dữ liệu TinixAI, ${source}. Phân tích bộ dữ liệu title đã lọc, không phải khảo sát toàn thị trường.`);
}
// 2 — cohort
{
 const s=slide('Dữ liệu và phạm vi phân tích',`Nguồn: ${source}; data/processed/tinixai_it_ai_data_jobs.metadata.json; reports/data_quality.json. CC BY-NC 4.0. Tỷ giá USD là giả định cấu hình.`);
 const salaryRows=Object.values(reg.split_rows).reduce((a,b)=>a+b,0)+reg.training_rows_removed;
 table(s,[['Bước','Số tin','Mục đích'],['TinixAI raw',number(provenance.source_rows),`Nguồn ${years[0].year}–${years.at(-1).year}`],['Title IT/AI/Data',number(quality.cleaned_rows),'EDA + phân cụm'],['Một trong 5 nghề',number(quality.classification_eligible_rows),'Phân loại nghề'],['Nghề đơn + midpoint',number(salaryRows),'Hồi quy lương']],162,[560,210,366],390);
 note(s,'Bộ lọc dựa trên title. Những tin thiếu lương hoặc nhãn nghề đơn có thể tạo sai lệch chọn mẫu.');
}
// 3 — year
{
 const s=slide('Phân bố năm trong bộ dữ liệu',`Nguồn: reports/tables/year_distribution.csv. year là năm được dataset cung cấp, không có ngày/tháng.`);
 bar(s,years.map(r=>r.year),[{name:'Số tin',values:years.map(r=>+r.jobs)}]);
 note(s,'Số dòng phụ thuộc độ phủ nguồn và tin đăng lại; năm 2026 có thể chưa đủ cả năm.');
}
// 4 — weak roles
{
 const s=slide('Nhóm nghề từ quy tắc title',`Nguồn: reports/tables/role_distribution.csv; src/preprocessing/role_normalizer.py. Other gồm nghề ngoài 5 nhóm và title nhiều nhóm/mơ hồ.`);
 bar(s,roles.map(r=>shortRole(r.role)),[{name:'Số tin',values:roles.map(r=>+r.jobs)}],{horizontal:true});
 note(s,`${number(quality.classification_eligible_rows)} tin có nhãn đơn. Nhãn từ title cần được kiểm định bằng người gán nhãn.`);
}
// 5 — skills
{
 const top=skills.slice(0,8);
 const s=slide('Kỹ năng được đề cập nhiều nhất',`Nguồn: reports/tables/skill_frequency.csv. Mỗi kỹ năng chỉ đếm tối đa một lần trong JD. FlashText có ranh giới từ Unicode; dữ liệu lấy từ mô tả/yêu cầu.`);
 bar(s,top.map(r=>r.skill),[{name:'Số JD',values:top.map(r=>+r.jobs)}],{horizontal:true});
 note(s,'Việc đề cập chưa phân biệt kỹ năng bắt buộc, ưu tiên hoặc phủ định.');
}
// 6 — salary distributions
{
 const rows=salaries.filter(r=>r.role_standard!=='Other');
 const s=slide('Trung vị midpoint lương theo nghề',`Nguồn: reports/tables/salary_by_role.csv. Tất cả midpoint hợp lệ trong dữ liệu, không cắt theo quantile. USD quy đổi ${quality.usd_to_vnd_rate} VND/USD, giả định cố định. Cỡ mẫu hiện trên nhãn.`);
 bar(s,rows.map(r=>`${shortRole(r.role_standard)} (n=${r.count})`),[{name:'Triệu VND/tháng',values:rows.map(r=>+r.median/1e6)}],{horizontal:true,format:'0.0'});
 note(s,'Đơn vị: triệu VND/tháng. Chưa điều chỉnh gross/net và lạm phát; đây là lương niêm yết.');
}
// 7 — evaluation protocol
{
 const s=slide('Đánh giá trên các công ty được giữ lại',`Nguồn: src/models/splitting.py; reports/classification_metrics.json; reports/salary_metrics.json. GroupShuffleSplit seed 42; khoảng 60/20/20 số công ty.`);
 table(s,[['Bài toán','Train','Validation','Test'],['Phân loại',...['train','validation','test'].map(k=>number(cls.split_rows[k]))],['Lương sau lọc train',...['train','validation','test'].map(k=>number(reg.split_rows[k]))]],166,[480,218,218,220],220);
 text(s,'TF-IDF, imputer, encoder và vocabulary model chỉ fit trên train.',72,425,1136,90,27);
 text(s,'Chọn model bằng validation; test dùng để báo cáo.',72,530,1136,62,27,true,C.teal);
 note(s,'Chia theo nhóm công ty để giảm trùng JD giữa tập; alias tên công ty vẫn cần chuẩn hóa thêm.');
}
// 8 — classifier
{
 const names=Object.keys(cls.models), labels=names.map(n=>n.replace('tfidf_',''));
 const s=slide('Phân loại nghề: TF-IDF + LinearSVC',`Nguồn: reports/classification_metrics.json. Model chọn ${cls.selected_model} bằng validation macro-F1. Baseline là lớp phổ biến trên train. Không dùng title trong đầu vào classifier.`);
 bar(s,labels,[{name:'Validation',values:names.map(n=>cls.models[n].validation.macro_f1)},
               {name:'Test',values:names.map(n=>cls.models[n].test.macro_f1)}],{legend:true,format:'0.000',max:1,h:390});
 text(s,`Macro-F1 test: ${metric(bestCls.macro_f1)}  ·  Baseline: ${metric(cls.baseline_macro_f1)}`,72,578,1136,46,27,true,C.teal);
 note(s,'F1 đo mức khớp nhãn từ quy tắc title; chưa phải độ chính xác so với nhãn chuyên gia.');
}
// 9 — salary models
{
 const names=Object.keys(reg.models);
 const s=slide('Dự đoán midpoint lương: LightGBM',`Nguồn: reports/salary_metrics.json. Chọn ${reg.selected_model} theo validation MAE. Học log1p(midpoint). Cắt đuôi train bằng quantile học từ train; giữ nguyên validation/test.`);
 bar(s,names.map(n=>n.replace('random_forest','Random Forest').replace('lightgbm','LightGBM').replace('ridge','Ridge')),
     [{name:'Validation MAE',values:names.map(n=>reg.models[n].validation.mae_vnd/1e6)},
      {name:'Test MAE',values:names.map(n=>reg.models[n].test.mae_vnd/1e6)}],{legend:true,format:'0.00',h:390});
 text(s,`Test: MAE ${million(bestReg.mae_vnd)} triệu  ·  R² ${metric(bestReg.r2)}`,72,578,1136,46,27,true,C.teal);
 note(s,`Baseline trung vị train: MAE ${million(reg.baseline_mae_vnd)} triệu VND/tháng. Cột biểu diễn MAE, càng thấp càng tốt.`);
}
// 10 — SHAP
{
 const labels={ 'categorical__seniority_Junior':'Seniority: Junior', 'numeric__experience_min':'Kinh nghiệm tối thiểu',
    'categorical__seniority_Unspecified':'Seniority: chưa xác định','numeric__skill_count':'Số kỹ năng',
    'categorical__job_position_Nhân viên':'Vị trí: Nhân viên','categorical__job_position_Nhân Viên':'Vị trí: Nhân Viên',
    'numeric__missingindicator_experience_min':'Thiếu kinh nghiệm','numeric__year':'Năm',
    'categorical__location_group_Hà Nội':'Địa điểm: Hà Nội'};
 const top=shap.slice(0,8);
 const s=slide('Các đặc trưng ảnh hưởng trong mô hình lương',`Nguồn: reports/salary_shap_importance.csv. Tree SHAP của ${reg.shap_model}, tối đa 150 mẫu test; thang log1p(lương).`);
 bar(s,top.map(r=>labels[r.feature]||r.feature.replace(/^[^_]+__/,'').replace('categorical__','')), [{name:'Mean |SHAP|',values:top.map(r=>+r.mean_abs_shap_log)}],{horizontal:true,format:'0.000'});
 note(s,'Mean |SHAP| ở thang log1p(lương). Không diễn giải thành mức tăng lương nhân quả của kỹ năng.');
}
// 11 — graph topology
{
 const top=centrality.slice(0,8);
 const s=slide('Mạng đồng xuất hiện kỹ năng',`Nguồn: reports/skill_centrality.csv; reports/clustering_metrics.json. Ngưỡng kỹ năng 50 tin, cạnh 20 tin; weighted degree là tổng count cạnh.`);
 bar(s,top.map(r=>r.skill),[{name:'Tổng count cạnh',values:top.map(r=>+r.weighted_degree)}],{horizontal:true,h:395});
 text(s,`${clustering.graph.nodes} kỹ năng  ·  ${number(clustering.graph.edges)} cạnh  ·  ${clustering.graph.components} thành phần liên thông`,72,582,1136,44,27,true,C.teal);
 note(s,'Cột: tổng count cạnh (weighted degree). Đồng xuất hiện chưa chứng minh phụ thuộc kỹ thuật.');
}
// 12 — semantic clustering
{
 const km=clustering.semantic.kmeans, hb=clustering.semantic.hdbscan;
 const s=slide('Phân cụm nội dung JD',`Nguồn: reports/clustering_metrics.json; configs/pipeline.json. Sentence model https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2. Embedding 384 chiều, chuẩn hóa, UMAP 15 chiều; seed 42. Silhouette trong không gian UMAP, HDBSCAN loại noise khi tính.`);
 table(s,[['Phương pháp','Số cụm','Noise','Silhouette'],['UMAP + KMeans',String(km.clusters),number(km.noise_rows),metric(km.silhouette_reduced_space)],['UMAP + HDBSCAN',String(hb.clusters),number(hb.noise_rows),hb.silhouette_reduced_space===null?'—':metric(hb.silhouette_reduced_space)]],174,[490,185,210,251],240);
 text(s,'Đọc title và JD trong từng cụm trước khi đặt tên.',72,463,1136,64,29,true,C.teal);
 text(s,'Cụm mô tả cấu trúc nội dung; chưa phải nhãn nghề đã kiểm định.',72,541,1136,64,27);
 note(s,'Embedding cắt JD ở 256 token; silhouette của hai phương pháp không dùng cùng tập điểm khi có noise.');
}
// 13 — conclusion
{
 const s=slide('Kết luận và bước tiếp theo',`Kết quả của lượt chạy trên ${quality.cleaned_rows} tin TinixAI. Nguồn: reports/project_report.md, classification_metrics.json, salary_metrics.json, clustering_metrics.json.`);
 text(s,`LinearSVC đạt macro-F1 ${metric(bestCls.macro_f1)} với nhãn title sơ bộ.`,72,177,1136,90,31,true,C.teal);
 text(s,`LightGBM có MAE ${million(bestReg.mae_vnd)} triệu VND/tháng trên test.`,72,285,1136,90,31,true,C.teal);
 text(s,'Bổ sung nhãn thủ công, chuẩn hóa công ty và đánh giá theo năm.',72,404,1136,90,29);
 text(s,'Chạy lại pipeline: python main.py --stage all',72,528,1136,72,28);
 note(s,'Báo cáo, notebook và artifacts có trong project; giới hạn nguồn và quy đổi lương cần được giữ khi trình bày.');
}

await fs.mkdir(build,{recursive:true});
await fs.mkdir(path.dirname(output),{recursive:true});
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:output,
  pythonExecutable:python,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',
              ...tables.flatMap(n=>['--require-native-table-slide',String(n)])],
  requiredNativeChartOwnerSlides:charts,requiredNativeTableOwnerSlides:tables,
  materializeLiteralChartWorkbooks:true,fontPolicy,verifyArtifactToolImport:true,
  receiptPath:path.join(build,path.basename(output)+'.validation.json')});
console.log('Final deck:',output);
// Render final file, not only the in-memory candidate.
const {FileBlob}=await import('@oai/artifact-tool');
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(output));
for(let i=0;i<finalDeck.slides.items.length;i++) {
 const blob=await finalDeck.export({slide:finalDeck.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
console.log('Rendered',finalDeck.slides.items.length,'slides; font:',family);
