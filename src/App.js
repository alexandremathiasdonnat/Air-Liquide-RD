import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from "recharts";
import "./App.css";
import { HMOE_REGIME_TYPES, ensureHmoeFeatures, runHmoe } from "./hmoe";
import { vnorm, runBOA, runMLpol, runMLprod, runFTRL } from "./moe";
import { formatDuration } from "./metrics";
import { estimateMonteCarloMs, runMonteCarloSimulation } from "./monteCarlo";
import { estimateMonteCarloGridSearchMs, runMonteCarloGridSearch } from "./monteCarloGridSearch";
import { buildConfiguredExperts as buildConfiguredExpertsData } from "./randomExperts";
import {
  buildAlgoRunLabel,
  cloneAlgoRunConfig,
  DEFAULT_EXTRA_PARAMS,
  DEFAULT_FTRL_PARAMS,
  DEFAULT_HMOE_REGIME_IDS,
  DEFAULT_LOSS_TYPE,
  DEFAULT_USE_GRAD,
  getHmoeRegimeNames,
  getHmoeRegimeSourceLabel,
  getMonteCarloAlgoParamTokens,
  getParamSourceLabel,
  resolveMonteCarloAlgoConfigs,
} from "./monteCarloConfig";
import {
  buildGridSearchComboLabel,
  createGridSearchCombo,
  getGridSearchComboDisplayTitle,
  getGridSearchComboSignature,
  getGridSearchControlSections,
  getInitialGridSearchComboOverrides,
} from "./gridSearchConfig";

// ─── Expert definitions ───────────────────────────────────────────────────────
const EXPERT_GROUPS = [
  { id:"bloc0", label:"Bloc 0 — Benchmarks", color:"#1a4da6", experts:[
    {id:"ridge_full", desc:"Modèle linéaire régularisé global. Benchmark stable, souvent lissant."},
    {id:"elasticnet_full", desc:"Modèle linéaire sparse global. Benchmark parcimonieux."},
    {id:"rf_full", desc:"Forêt aléatoire globale non linéaire. Benchmark robuste."},
    {id:"lgbm_full", desc:"Boosting non linéaire global. Souvent le meilleur benchmark."},
  ]},
  { id:"specialists", label:"Spécialistes", color:"#0a7a52", experts:[
    {id:"short_horizon_specialist", desc:"Expert horizons courts uniquement."},
    {id:"long_horizon_specialist", desc:"Expert horizons longs, prévisions lointaines."},
    {id:"late_vector_specialist", desc:"Expert fin de vecteur day-ahead, 2e moitié de journée."},
    {id:"strong_wind_specialist", desc:"Expert vent fort, entraîné sur observations à fort vent."},
    {id:"low_wind_specialist", desc:"Expert vent faible, seuils bas et faible production."},
    {id:"gusty_regime_specialist", desc:"Expert régimes rafaleux, conditions instables."},
    {id:"stable_wind_specialist", desc:"Expert vents réguliers, régimes peu turbulents."},
    {id:"night_specialist", desc:"Expert heures de nuit."},
    {id:"day_specialist", desc:"Expert heures de jour."},
    {id:"winter_specialist", desc:"Expert hiver, structure d'erreur hivernale."},
    {id:"summer_specialist", desc:"Expert été, structure saisonnière estivale."},
  ]},
  { id:"derived", label:"Dérivés", color:"#a01860", experts:[
    {id:"rf_drift_down_after_midpoint", desc:"Dérivé RF. Se dégrade après une date."},
    {id:"lgbm_drift_up_after_midpoint", desc:"Dérivé LGBM. S'améliore après une date."},
    {id:"ridge_break_after_date", desc:"Dérivé Ridge. Casse brutale après une date seuil."},
    {id:"lgbm_peak_underestimator", desc:"Dérivé LGBM. Sous-estime les pics."},
    {id:"ridge_smoother", desc:"Dérivé Ridge. Lisse excessivement."},
    {id:"rf_slow_reactor", desc:"Dérivé RF. Réagit lentement aux changements."},
    {id:"low_value_overestimator", desc:"Dérivé Ridge/RF. Surestime les faibles valeurs."},
    {id:"ridge_biased_low", desc:"Dérivé Ridge. Sous-prédit globalement."},
    {id:"rf_biased_high", desc:"Dérivé RF. Sur-prédit globalement."},
    {id:"lgbm_amplitude_compressed", desc:"Dérivé LGBM. Compresse l'amplitude."},
    {id:"elasticnet_additive_bias", desc:"Dérivé ElasticNet. Biais constant additif."},
  ]},
  { id:"restricted", label:"Features restreintes", color:"#8a6200", experts:[
    {id:"wind_only_expert", desc:"N'utilise que le vent."},
    {id:"history_horizon_expert", desc:"N'utilise que l'historique et l'horizon."},
    {id:"no_lag_expert", desc:"Ignore l'historique de production."},
    {id:"no_cloud_pressure_expert", desc:"Ignore cloud/pressure/humidity."},
  ]},
];
const ALL_SYNTHETIC = EXPERT_GROUPS.flatMap(g=>g.experts.map(e=>e.id));

const ALGO_GROUPS = [
  { label:"Opera - MOE", algos:[
    {id:"BOA",name:"MOE BOA",desc:"Bernstein Online Aggregation.",params:[]},
    {id:"MLpol",name:"MOE MLpol",desc:"Multiplicative Weights Polynomial.",params:[]},
    {id:"MLprod",name:"MOE MLprod",desc:"Multiplicative Weights Prod.",params:[]},
    {id:"FTRL",name:"MOE FTRL",desc:"Follow The Regularized Leader.",params:[
      {id:"eta0",label:"Learning rate η₀",type:"slider",min:0.001,max:0.5,step:0.001,default:0.01},
      {id:"tol",label:"Tolérance",type:"select",options:[1e-5,1e-10,1e-15,1e-20],default:1e-20},
      {id:"maxiter",label:"Max itérations",type:"slider",min:10,max:200,step:10,default:50},
    ]},
  ]},
  { label:"Opera - HMOE", algos:[
    {id:"HMOE_BOA",name:"HMOE BOA",desc:"BOA avec branches regime-gated HMOE.",params:[]},
    {id:"HMOE_MLpol",name:"HMOE MLpol",desc:"MLpol avec branches regime-gated HMOE.",params:[]},
    {id:"HMOE_MLprod",name:"HMOE MLprod",desc:"MLprod avec branches regime-gated HMOE.",params:[]},
    {id:"HMOE_FTRL",name:"HMOE FTRL",desc:"FTRL avec branches regime-gated HMOE.",params:[
      {id:"eta0",label:"Learning rate η₀",type:"slider",min:0.001,max:0.5,step:0.001,default:0.01},
      {id:"tol",label:"Tolérance",type:"select",options:[1e-5,1e-10,1e-15,1e-20],default:1e-20},
      {id:"maxiter",label:"Max itérations",type:"slider",min:10,max:200,step:10,default:50},
    ]},
  ]},
  { label:"Statiques", algos:[
    {id:"SimpleMean",name:"Moyenne simple",desc:"Moyenne arithmétique non pondérée.",params:[]},
    {id:"Median",name:"Médiane",desc:"Médiane des prédictions.",params:[]},
    {id:"TrimmedMean",name:"Trimmed Mean",desc:"Moyenne après exclusion des X% d'experts.",params:[
      {id:"trim",label:"Trim (%)",type:"slider",min:5,max:40,step:5,default:20},
    ]},
  ]},
  { label:"Adaptatifs", algos:[
    {id:"InvMSE",name:"Inverse MSE",desc:"Poids inversement proportionnels au MSE.",params:[
      {id:"window",label:"Fenêtre (pas)",type:"slider",min:6,max:168,step:6,default:48},
    ]},
    {id:"BestExpert",name:"Best Expert",desc:"Sélectionne l'expert avec la plus faible MAE.",params:[
      {id:"window",label:"Fenêtre (pas)",type:"slider",min:6,max:168,step:6,default:48},
    ]},
    {id:"Ridge",name:"Ridge Blending",desc:"Combinaison linéaire régularisée L2.",params:[
      {id:"alpha",label:"Régularisation α",type:"slider",min:0.1,max:50,step:0.1,default:1},
    ]},
  ]},
];
const ALGOS = ALGO_GROUPS.flatMap(g=>g.algos);
const OPERA_ALGO_IDS = ["BOA","MLpol","MLprod","FTRL"];
const HMOE_ALGO_IDS = ["HMOE_BOA","HMOE_MLpol","HMOE_MLprod","HMOE_FTRL"];
const LOSS_TYPES = [{id:"mse",label:"MSE"},{id:"mae",label:"MAE"},{id:"mape",label:"MAPE"},{id:"msle",label:"MSLE"},{id:"mspe",label:"MSPE"}];
const PALETTE = ["#f0256a","#2b7fff","#00c176","#f5a800","#8b5cf6","#ff6a20","#ff1f5a","#00bcd4","#7ecb00","#ff8c00"];
const MOE_PALETTE = ["#ff4d8f","#4d9fff","#00d97e","#ffbe2e","#a374f7","#ff7c35","#ff4d70","#26d0e6","#93d800","#ffa040"];
const THEME = {
  appBg:"#153f9d",
  panelBg:"#c8ddf5",
  panelBgSoft:"#b8cfed",
  border:"#9ab8d8",
  textPrimary:"#0e2d52",
  textSecondary:"#1a3f6a",
  textMuted:"#1e4e7a",
  textDim:"#25608a",
  grid:"#8fb5d8",
};

// ─── Math helpers (statiques/adaptatifs) ─────────────────────────────────────
function solveLinear(A,b){const n=b.length,M=A.map((row,i)=>[...row,b[i]]);for(let col=0;col<n;col++){let max=col;for(let row=col+1;row<n;row++)if(Math.abs(M[row][col])>Math.abs(M[max][col]))max=row;[M[col],M[max]]=[M[max],M[col]];for(let row=col+1;row<n;row++){const f=M[row][col]/M[col][col];for(let j=col;j<=n;j++)M[row][j]-=f*M[col][j];}}const x=new Array(n).fill(0);for(let i=n-1;i>=0;i--){x[i]=M[i][n];for(let j=i+1;j<n;j++)x[i]-=M[i][j]*x[j];x[i]/=M[i][i];}return x;}
function runSimpleMean(data,cols){const K=cols.length;return{predictions:data.map(r=>cols.reduce((s,c)=>s+(r[c]||0),0)/K),weightHistory:data.map(()=>new Array(K).fill(1/K))};}
function runMedian(data,cols){const K=cols.length,preds=[],wh=[];for(let t=0;t<data.length;t++){const vals=cols.map(c=>data[t][c]||0),sorted=[...vals].sort((a,b)=>a-b);const med=K%2===0?(sorted[K/2-1]+sorted[K/2])/2:sorted[Math.floor(K/2)];const dists=vals.map(v=>Math.abs(v-med)+1e-8);preds.push(med);wh.push(vnorm(dists.map(d=>1/d)));}return{predictions:preds,weightHistory:wh};}
function runTrimmedMean(data,cols,params){const K=cols.length,nT=Math.max(0,Math.floor(K*(params.trim||20)/100/2));const preds=[],wh=[];for(let t=0;t<data.length;t++){const vals=cols.map(c=>data[t][c]||0);const idxSorted=vals.map((v,i)=>({v,i})).sort((a,b)=>a.v-b.v);const kept=idxSorted.slice(nT,K-nT);const pred=kept.reduce((s,x)=>s+x.v,0)/kept.length;const w=new Array(K).fill(0);kept.forEach(x=>{w[x.i]=1/kept.length;});preds.push(pred);wh.push(w);}return{predictions:preds,weightHistory:wh};}
function runInvMSE(data,cols,params){const K=cols.length,win=params.window||48,preds=[],wh=[];for(let t=0;t<data.length;t++){const x=cols.map(c=>data[t][c]||0);let w;if(t<2){w=new Array(K).fill(1/K);}else{const sl=data.slice(Math.max(0,t-win),t);const mses=cols.map(c=>{const e=sl.map(r=>(r[c]||0)-r.y_true);return e.reduce((s,v)=>s+v**2,0)/sl.length+1e-6;});w=vnorm(mses.map(m=>1/m));}preds.push(w.reduce((s,wk,k)=>s+wk*x[k],0));wh.push([...w]);}return{predictions:preds,weightHistory:wh};}
function runBestExpert(data,cols,params){const K=cols.length,win=params.window||48,preds=[],wh=[];for(let t=0;t<data.length;t++){const x=cols.map(c=>data[t][c]||0);let bi=0;if(t>=2){const sl=data.slice(Math.max(0,t-win),t);const maes=cols.map(c=>{const e=sl.map(r=>Math.abs((r[c]||0)-r.y_true));return e.reduce((s,v)=>s+v,0)/sl.length;});bi=maes.indexOf(Math.min(...maes));}const w=new Array(K).fill(0);w[bi]=1;preds.push(x[bi]);wh.push([...w]);}return{predictions:preds,weightHistory:wh};}
function runRidge(data,cols,params){const K=cols.length,alpha=params.alpha||1;const X=data.map(r=>cols.map(c=>r[c]||0)),y=data.map(r=>r.y_true);const XtX=Array.from({length:K},(_,i)=>Array.from({length:K},(_,j)=>X.reduce((s,row)=>s+row[i]*row[j],0)+(i===j?alpha:0)));const Xty=Array.from({length:K},(_,i)=>X.reduce((s,row,t)=>s+row[i]*y[t],0));const wR=solveLinear(XtX,Xty);const wD=vnorm(wR.map(Math.abs));return{predictions:X.map(row=>row.reduce((s,v,k)=>s+v*wR[k],0)),weightHistory:data.map(()=>[...wD])};}
function runAlgo(data,cols,algoId,lt,ug,ep,fp){switch(algoId){case"BOA":return runBOA(data,cols,lt,ug);case"MLpol":return runMLpol(data,cols,lt,ug);case"MLprod":return runMLprod(data,cols,lt,ug);case"FTRL":return runFTRL(data,cols,lt,ug,fp);case"SimpleMean":return runSimpleMean(data,cols);case"Median":return runMedian(data,cols);case"TrimmedMean":return runTrimmedMean(data,cols,ep);case"InvMSE":return runInvMSE(data,cols,ep);case"BestExpert":return runBestExpert(data,cols,ep);case"Ridge":return runRidge(data,cols,ep);default:return runBOA(data,cols,lt,ug);}}
function getHmoeBaseAlgoId(algoId){return algoId.startsWith("HMOE_")?algoId.replace("HMOE_",""):algoId;}

// ─── Rand Expert generation ───────────────────────────────────────────────────
function randInt(a,b){return Math.floor(Math.random()*(b-a+1))+a;}
function addNoise(val,noiseLevel){return val*(1+(Math.random()*2-1)*noiseLevel);}

function generateRandExperts(rows, nExperts, phaseRange, noiseLevel){
  const n=rows.length;
  const experts=[];
  for(let e=0;e<nExperts;e++){
    const nPhases=randInt(phaseRange[0],phaseRange[1]);
    // Random breakpoints
    const breaks=[0];
    const pts=[];
    for(let p=0;p<nPhases-1;p++) pts.push(randInt(1,n-1));
    pts.sort((a,b)=>a-b);
    const uniquePts=[...new Set(pts)];
    breaks.push(...uniquePts,n);
    const phases=[];
    for(let p=0;p<breaks.length-1;p++){
      const expert=ALL_SYNTHETIC[randInt(0,ALL_SYNTHETIC.length-1)];
      phases.push({start:breaks[p],end:breaks[p+1],expert});
    }
    const colId=`rand_expert_${e+1}`;
    const values=rows.map((row,t)=>{
      const phase=phases.find(ph=>t>=ph.start&&t<ph.end)||phases[phases.length-1];
      const base=row[phase.expert]||0;
      return noiseLevel>0?addNoise(base,noiseLevel):base;
    });
    experts.push({id:colId,label:`R-Expert ${e+1}`,phases,values,noiseLevel});
  }
  return experts;
}

function buildDataWithRandExperts(rows, randExperts){
  return rows.map((row,t)=>{
    const o={...row};
    randExperts.forEach(re=>{o[re.id]=re.values[t];});
    return o;
  });
}

// ─── CSV Parser ───────────────────────────────────────────────────────────────
function parseCSV(text){
  const clean=text.replace(/^\uFEFF/,"").replace(/\r\n/g,"\n").replace(/\r/g,"\n");
  const lines=clean.trim().split("\n").filter(l=>l.trim()!=="");
  const headers=lines[0].split(",").map(h=>h.trim().replace(/^"|"$/g,""));
  return lines.slice(1).map(line=>{
    const vals=[];let cur="",inQ=false;
    for(let i=0;i<line.length;i++){const c=line[i];if(c==='"')inQ=!inQ;else if(c===','&&!inQ){vals.push(cur.trim());cur="";}else cur+=c;}
    vals.push(cur.trim());
    const obj={};headers.forEach((h,i)=>{obj[h]=(vals[i]??"").replace(/^"|"$/g,"").trim();});
    return obj;
  });
}

const DEMO_CSV=`decision_time,target_time,horizon,y_true,ridge_full,elasticnet_full,rf_full,lgbm_full,short_horizon_specialist,long_horizon_specialist,late_vector_specialist,strong_wind_specialist,low_wind_specialist,gusty_regime_specialist,stable_wind_specialist,night_specialist,day_specialist,winter_specialist,summer_specialist,wind_only_expert,history_horizon_expert,no_lag_expert,no_cloud_pressure_expert,rf_drift_down_after_midpoint,lgbm_drift_up_after_midpoint,ridge_break_after_date,lgbm_peak_underestimator,ridge_smoother,rf_slow_reactor,low_value_overestimator,ridge_biased_low,rf_biased_high,lgbm_amplitude_compressed,elasticnet_additive_bias
2025-02-09 10:00:00+00:00,2025-02-10 00:00:00+00:00,14,930.13,1005.63,1003.01,919.37,765.42,862.46,1097.48,1022.73,1869.06,215.61,906.50,923.29,774.24,1230.97,923.29,923.29,784.82,762.54,800.05,925.54,919.37,727.15,1005.63,765.42,974.41,919.37,919.37,905.07,1011.31,775.77,1053.01
2025-02-09 10:00:00+00:00,2025-02-10 01:00:00+00:00,15,977.59,928.23,925.94,964.43,896.54,886.25,1135.72,1017.71,1893.78,239.42,794.72,962.04,775.54,1198.50,962.04,962.04,786.92,1292.84,796.51,957.11,964.43,851.72,928.23,896.54,1015.39,932.89,964.43,835.41,1060.88,887.22,975.94
2025-02-09 10:00:00+00:00,2025-02-10 02:00:00+00:00,16,1314.43,989.38,988.08,943.87,913.18,897.93,1137.59,1057.24,1875.55,214.88,841.66,943.03,786.17,1192.81,943.03,943.03,780.39,1363.82,780.57,972.11,943.87,867.52,989.38,913.18,1067.78,936.19,943.87,890.44,1038.26,901.36,1038.08
2025-02-09 10:00:00+00:00,2025-02-10 03:00:00+00:00,17,1505.03,1138.31,1139.36,1095.25,1263.96,964.16,1264.79,1266.27,1896.75,229.11,904.94,1100.18,914.93,1373.83,1100.18,1100.18,1149.02,1574.87,1083.18,1158.43,1095.25,1200.76,1138.31,1263.96,1100.30,983.90,1095.25,1024.48,1204.77,1199.53,1189.36
2025-02-09 10:00:00+00:00,2025-02-10 04:00:00+00:00,18,1658.4,1277.37,1279.38,1273.49,1465.83,1199.65,1653.95,1564.18,1820.15,226.05,1092.09,1281.61,1136.02,1487.86,1281.61,1281.61,1375.60,1703.85,1343.71,1313.78,1273.49,1392.54,1277.37,1465.83,1162.02,1070.78,1273.49,1149.63,1400.84,1371.12,1329.38
2025-02-09 10:00:00+00:00,2025-02-10 05:00:00+00:00,19,1621.47,1168.20,1167.71,1197.63,1262.46,958.21,1330.39,1349.53,1868.08,190.58,1163.12,1199.56,896.27,1381.03,1199.56,1199.56,1251.95,1783.52,1236.39,1224.47,1197.63,1199.34,1168.20,1262.46,1209.88,1108.84,1197.63,1051.38,1317.40,1198.25,1217.71
2025-02-09 10:00:00+00:00,2025-02-10 06:00:00+00:00,20,1590.5,1236.82,1235.92,1253.88,1322.59,982.27,1341.15,1413.18,1866.76,197.75,1091.44,1243.97,947.37,1360.57,1243.97,1243.97,1313.01,1825.68,1235.56,1313.68,1253.88,1256.46,1236.82,1322.59,1199.42,1152.35,1253.88,1113.14,1379.27,1249.36,1285.92
2025-02-09 10:00:00+00:00,2025-02-10 07:00:00+00:00,21,1666.66,1228.71,1227.68,1286.93,1391.71,1003.40,1345.05,1391.53,1853.95,189.54,1078.02,1282.75,953.40,1371.30,1282.75,1282.75,1377.26,1846.22,1300.49,1327.92,1286.93,1322.13,1228.71,1391.71,1243.34,1192.72,1286.93,1105.84,1415.62,1308.12,1277.68
2025-02-09 10:00:00+00:00,2025-02-10 08:00:00+00:00,22,1494.94,1086.01,1082.53,993.40,1159.26,934.18,1161.76,1145.52,1869.49,182.77,1068.64,1002.74,830.45,1236.63,1002.74,1002.74,916.98,1950.20,888.34,1020.62,993.40,1101.30,1086.01,1159.26,1279.30,1132.93,993.40,977.41,1092.74,1110.53,1132.53
2025-02-09 10:00:00+00:00,2025-02-10 09:00:00+00:00,23,1239.99,1496.95,1498.40,1495.06,1651.99,1268.17,1734.32,1616.54,1946.18,201.14,1235.98,1487.65,1199.01,1618.75,1487.65,1487.65,1635.90,1992.35,1480.86,1553.78,1495.06,1569.39,1496.95,1404.19,1333.19,1241.57,1495.06,1347.26,1644.57,1529.35,1548.40
2025-02-09 10:00:00+00:00,2025-02-10 10:00:00+00:00,24,1503.51,1348.00,1346.14,1274.01,1209.68,1015.57,1349.06,1405.15,1879.05,199.04,1077.10,1274.30,961.38,1375.10,1274.30,1274.30,1412.87,1978.81,1237.90,1379.29,1274.01,1149.19,1348.00,1209.68,1392.33,1251.30,1274.01,1213.20,1401.41,1153.39,1396.14
2025-02-09 10:00:00+00:00,2025-02-10 11:00:00+00:00,25,1683.97,1506.27,1505.86,1484.46,1683.41,1210.11,1698.37,1623.77,1875.70,204.67,1180.13,1488.33,1152.73,1545.33,1488.33,1488.33,1627.86,1989.61,1493.20,1562.37,1484.46,1599.24,1506.27,1430.90,1480.16,1321.25,1484.46,1355.64,1632.90,1556.06,1555.86
2025-02-09 10:00:00+00:00,2025-02-10 12:00:00+00:00,26,1725.19,1524.40,1522.84,1527.82,1740.06,1213.11,1704.94,1638.50,1873.13,207.94,1197.26,1532.69,1191.53,1576.30,1532.69,1532.69,1654.72,2004.11,1523.13,1592.77,1527.82,1653.05,1524.40,1479.05,1494.10,1383.22,1527.82,1371.96,1680.60,1604.21,1572.84
2025-02-09 10:00:00+00:00,2025-02-10 13:00:00+00:00,27,1590.97,1525.17,1522.84,1567.74,1712.32,1170.22,1693.81,1616.06,1873.51,228.79,1149.40,1557.86,1104.24,1527.95,1557.86,1557.86,1696.64,2003.13,1554.72,1620.06,1567.74,1626.70,1525.17,1455.47,1547.19,1438.57,1567.74,1372.65,1724.52,1580.63,1572.84
2025-02-09 10:00:00+00:00,2025-02-10 14:00:00+00:00,28,1613.44,1566.66,1563.69,1631.79,1781.47,1217.41,1727.79,1653.36,1879.45,226.30,1222.81,1627.57,1193.70,1616.97,1627.57,1627.57,1736.40,2021.43,1615.87,1666.19,1631.79,1692.40,1566.66,1514.25,1569.57,1496.54,1631.79,1409.99,1794.97,1639.41,1613.69
2025-02-09 10:00:00+00:00,2025-02-10 15:00:00+00:00,29,1720.6,1613.44,1613.32,1737.14,1767.31,1402.23,1782.96,1677.54,1894.85,216.55,1314.99,1730.59,1318.84,1685.92,1730.59,1730.59,1782.91,2029.51,1655.78,1768.48,1737.14,1678.95,1613.44,1502.21,1596.95,1568.72,1737.14,1452.10,1910.85,1627.37,1663.32
2025-02-09 10:00:00+00:00,2025-02-10 16:00:00+00:00,30,1829.31,1618.19,1615.51,1747.88,1822.45,1235.61,1743.81,1666.25,1957.08,229.83,1242.17,1740.08,1182.59,1590.66,1740.08,1740.08,1801.27,2019.02,1702.69,1778.49,1747.88,1731.32,1618.19,1549.08,1605.51,1622.47,1747.88,1456.37,1922.66,1674.24,1665.51
2025-02-09 10:00:00+00:00,2025-02-10 17:00:00+00:00,31,1860.47,1661.28,1659.20,1742.87,1779.65,1456.92,1823.26,1706.51,1970.61,229.58,1381.30,1734.26,1548.09,1722.99,1734.26,1734.26,1796.36,1939.15,1673.26,1765.22,1742.87,1690.67,1661.28,1512.70,1578.88,1658.59,1742.87,1495.15,1917.16,1637.86,1709.20
2025-02-09 10:00:00+00:00,2025-02-10 18:00:00+00:00,32,1828.12,1567.98,1563.64,1736.33,1775.10,1217.82,1769.36,1668.86,1974.49,230.37,1239.12,1733.15,1189.15,1614.18,1733.15,1733.15,1808.01,1901.27,1693.95,1769.88,1736.33,1686.34,1567.98,1508.83,1531.79,1681.91,1736.33,1411.18,1909.97,1633.99,1613.64
2025-02-09 10:00:00+00:00,2025-02-10 19:00:00+00:00,33,1575.87,1433.54,1428.02,1439.16,1354.87,1032.49,1492.78,1479.16,1970.32,231.61,1216.34,1438.59,981.01,1452.38,1438.59,1438.59,1479.27,1820.95,1388.32,1443.90,1439.16,1287.12,1433.54,1354.87,1470.04,1609.08,1439.16,1290.18,1583.07,1276.80,1478.02
2025-02-09 10:00:00+00:00,2025-02-10 20:00:00+00:00,34,1338.89,1377.97,1371.11,959.99,975.17,1004.51,1326.16,1214.78,1905.51,213.59,1054.29,956.40,916.09,1439.50,956.40,956.40,1025.10,1712.63,950.61,975.33,959.99,926.41,1377.97,975.17,1380.71,1414.36,959.99,1240.18,1055.99,954.06,1421.11
2025-02-09 10:00:00+00:00,2025-02-10 21:00:00+00:00,35,1183.38,1309.42,1302.79,852.16,938.55,937.21,1195.49,1133.75,1888.51,212.39,926.03,844.29,877.35,1339.19,844.29,844.29,930.32,1648.01,827.30,874.03,852.16,891.62,1309.42,938.55,1243.05,1245.70,852.16,1178.48,937.38,922.92,1352.79
2025-02-09 10:00:00+00:00,2025-02-10 22:00:00+00:00,36,820.67,1214.66,1205.58,929.03,816.28,963.01,1040.86,922.32,1875.42,235.65,901.20,933.57,869.16,1310.08,933.57,933.57,842.47,1594.16,840.58,897.07,929.03,775.46,1214.66,816.28,1093.48,1150.70,929.03,1093.19,1021.93,818.99,1255.58
2025-02-09 10:00:00+00:00,2025-02-10 23:00:00+00:00,37,601.77,879.68,868.11,452.22,663.49,607.63,479.96,481.42,1890.03,223.63,415.45,450.27,630.73,551.45,450.27,450.27,505.73,1516.30,450.84,428.71,452.22,630.31,879.68,663.49,922.03,941.15,452.22,791.71,497.44,689.12,918.11
2025-02-10 10:00:00+00:00,2025-02-11 00:00:00+00:00,14,571.88,685.66,674.77,301.00,428.39,335.01,361.44,348.38,1884.27,200.16,273.39,304.06,374.53,377.88,304.06,304.06,468.67,1512.08,330.73,293.36,301.00,406.97,685.66,428.39,798.04,749.11,301.00,617.09,331.11,489.29,724.77
2025-02-10 10:00:00+00:00,2025-02-11 01:00:00+00:00,15,653.07,520.73,512.99,333.85,504.61,358.28,375.49,355.42,1909.31,214.17,297.50,333.47,389.98,496.86,333.47,333.47,483.39,1416.13,352.38,320.24,333.85,479.38,520.73,504.61,644.17,624.53,333.85,468.66,367.23,554.08,562.99
2025-02-10 10:00:00+00:00,2025-02-11 02:00:00+00:00,16,419.33,689.47,681.47,405.21,580.83,397.13,498.63,489.19,1896.13,212.15,420.60,402.01,489.44,555.29,402.01,402.01,542.14,1084.19,415.48,402.22,405.21,551.79,689.47,580.83,530.77,558.73,405.21,620.52,445.73,618.87,731.47`;

// ─── Double Range Slider ──────────────────────────────────────────────────────
function DoubleSlider({min,max,valMin,valMax,onChangeMin,onChangeMax,color="#a78bfa"}){
  const trackRef=useRef();
  const dragging=useRef(null); // "min" | "max" | null
  const steps=max-min;
  const pctMin=((valMin-min)/steps)*100;
  const pctMax=((valMax-min)/steps)*100;
  const ticks=Array.from({length:steps+1},(_,i)=>i+min);

  const pctFromEvent=useCallback(e=>{
    const rect=trackRef.current.getBoundingClientRect();
    const clientX=e.touches?e.touches[0].clientX:e.clientX;
    const raw=(clientX-rect.left)/rect.width;
    const clamped=Math.max(0,Math.min(1,raw));
    return Math.round(clamped*steps)+min;
  },[steps,min]);

  const onMouseDown=(thumb,e)=>{
    e.preventDefault();
    dragging.current=thumb;
    const move=ev=>{
      const v=pctFromEvent(ev);
      if(dragging.current==="min"&&v<=valMax) onChangeMin(v);
      if(dragging.current==="max"&&v>=valMin) onChangeMax(v);
    };
    const up=()=>{dragging.current=null;window.removeEventListener("mousemove",move);window.removeEventListener("mouseup",up);window.removeEventListener("touchmove",move);window.removeEventListener("touchend",up);};
    window.addEventListener("mousemove",move);
    window.addEventListener("mouseup",up);
    window.addEventListener("touchmove",move,{passive:false});
    window.addEventListener("touchend",up);
  };

  /*
  const gridSearchSelectedGroup=ALGO_GROUPS.find(group=>group.label===gridSearchGroupLabel)||ALGO_GROUPS[0];
  const gridSearchAvailableAlgos=gridSearchSelectedGroup?.algos||[];
  const gridSearchSelectedAlgo=ALGOS.find(algo=>algo.id===gridSearchAlgoId)||gridSearchAvailableAlgos[0]||ALGOS[0];
  const gridSearchControlSections=useMemo(
    ()=>getGridSearchControlSections(gridSearchAlgoId,LOSS_TYPES,HMOE_REGIME_TYPES),
    [gridSearchAlgoId],
  );
  const gridSearchDistinctCount=useMemo(
    ()=>new Set(gridSearchCombos.map(combo=>getGridSearchComboSignature(gridSearchAlgoId,combo))).size,
    [gridSearchAlgoId,gridSearchCombos],
  );
  const gridSearchEstimateMs=useMemo(()=>{
    if(!lastRandomSetup||!gridSearchAlgoId||gridSearchSimulationCount<1||gridSearchCombos.length===0)return 0;
    return estimateMonteCarloGridSearchMs({
      rowCount:lastRandomSetup.rowCount,
      simulationCount:gridSearchSimulationCount,
      randomConfig:lastRandomSetup,
      algoId:gridSearchAlgoId,
      comboCount:gridSearchCombos.length,
    });
  },[lastRandomSetup,gridSearchSimulationCount,gridSearchAlgoId,gridSearchCombos.length]);
  const gridSearchCanRun=expertMode==="random"
    &&!!lastRandomSetup
    &&lastRandomSetup.rows.length>=2
    &&gridSearchSimulationCount>=3
    &&gridSearchCombos.length>=2
    &&gridSearchDistinctCount>=2
    &&(!HMOE_ALGO_IDS.includes(gridSearchAlgoId)||gridSearchCombos.every(combo=>combo.selectedHmoeRegimes.length>=1))
    &&!gridSearchState.running;
  const gridSearchWarnings=[
    !prodMode&&expertMode!=="random"?"Le gridsearch Monte Carlo n'est disponible que si le mode Aléatoire est actif.":null,
    !prodMode&&!lastRandomSetup?"Cliquez au moins une fois sur « Générer X experts » pour figer les conditions de génération de référence.":null,
    gridSearchSimulationCount<3?"Le nombre de simulations doit être un entier n > 2.":null,
    gridSearchCombos.length<2?"Ajoutez au moins 2 combinaisons de paramètres.":null,
    gridSearchCombos.length>=2&&gridSearchDistinctCount<2?"Les combinaisons doivent être distinctes pour permettre le classement.":null,
    HMOE_ALGO_IDS.includes(gridSearchAlgoId)&&gridSearchCombos.some(combo=>combo.selectedHmoeRegimes.length<1)?"Chaque combinaison HMOE doit conserver au moins 1 régime.":null,
  ].filter(Boolean);
  const gridSearchCurrentLabel=gridSearchState.currentComboLabel;
  const handleStopGridSearch=()=>{
    if(!gridSearchAbortRef.current)return;
    gridSearchAbortRef.current.abort();
    setGridSearchState(prev=>({...prev,stage:"cancelling",cancelRequested:true,error:null}));
  };
  const handleGridSearchGroupChange=nextGroupLabel=>{
    const nextGroup=ALGO_GROUPS.find(group=>group.label===nextGroupLabel)||ALGO_GROUPS[0];
    const nextAlgoId=nextGroup?.algos?.[0]?.id||"BOA";
    setGridSearchGroupLabel(nextGroup.label);
    setGridSearchAlgoId(nextAlgoId);
    setGridSearchCombos(buildGridSearchCombosForAlgo(nextAlgoId));
    setGridSearchResult(null);
    setGridSearchState(createGridSearchAsyncState());
  };
  const handleGridSearchAlgoChange=nextAlgoId=>{
    setGridSearchAlgoId(nextAlgoId);
    setGridSearchCombos(buildGridSearchCombosForAlgo(nextAlgoId));
    setGridSearchResult(null);
    setGridSearchState(createGridSearchAsyncState());
  };
  const updateGridSearchCombo=(comboId,updater)=>{
    setGridSearchCombos(prev=>prev.map(combo=>combo.id===comboId?updater(combo):combo));
  };
  const updateGridSearchControl=(comboId,scope,field,value)=>{
    updateGridSearchCombo(comboId,combo=>{
      if(scope==="root")return{...combo,[field]:value};
      return{...combo,[scope]:{...combo[scope],[field]:value}};
    });
  };
  const toggleGridSearchRegime=(comboId,regimeId)=>{
    updateGridSearchCombo(comboId,combo=>{
      const selected=combo.selectedHmoeRegimes.includes(regimeId);
      const nextSelected=selected
        ?(combo.selectedHmoeRegimes.length>1?combo.selectedHmoeRegimes.filter(id=>id!==regimeId):combo.selectedHmoeRegimes)
        :[...combo.selectedHmoeRegimes,regimeId];
      return{...combo,selectedHmoeRegimes:nextSelected};
    });
  };
  const addGridSearchCombo=()=>{
    setGridSearchCombos(prev=>{
      const lastCombo=prev[prev.length-1];
      if(!lastCombo)return buildGridSearchCombosForAlgo(gridSearchAlgoId);
      const {id:_ignored,...comboSeed}=lastCombo;
      return[...prev,createGridSearchCombo(gridSearchAlgoId,gridSearchComboIdRef.current++,comboSeed)];
    });
  };
  const removeGridSearchCombo=comboId=>{
    setGridSearchCombos(prev=>prev.length>1?prev.filter(combo=>combo.id!==comboId):prev);
  };
  const handleGridSearchRun=async()=>{
    if(!gridSearchCanRun||!lastRandomSetup)return;
    const frozenRandomSetup=lastRandomSetup;
    const combos=gridSearchCombos.map((combo,index)=>({
      ...combo,
      extraP:{...combo.extraP},
      ftrlP:{...combo.ftrlP},
      selectedHmoeRegimes:[...combo.selectedHmoeRegimes],
      label:buildGridSearchComboLabel(gridSearchAlgoId,combo,index),
    }));
    const estimatedMs=gridSearchEstimateMs;
    const controller=new AbortController();
    gridSearchAbortRef.current=controller;
    const controller=new AbortController();
    gridSearchAbortRef.current=controller;
    setTab("gridsearchmc");
    setGridSearchResult(null);
    setGridSearchState({running:true,progress:0,elapsedMs:0,remainingMs:estimatedMs,stage:"starting",simulationIndex:0,comboIndex:-1,currentComboLabel:null,error:null,cancelRequested:false});
    try{
      const result=await runMonteCarloGridSearch({
        rows:frozenRandomSetup.rows,
        simulationCount:gridSearchSimulationCount,
        algoId:gridSearchAlgoId,
        combos,
        randomConfig:frozenRandomSetup,
        syntheticIds:ALL_SYNTHETIC,
        signal:controller.signal,
        onProgress:progress=>{
          setGridSearchState(prev=>({...prev,...progress,running:progress.stage!=="done",error:null}));
        },
      });
      if(gridSearchAbortRef.current===controller)gridSearchAbortRef.current=null;
      setGridSearchResult({...result,randomSetup:frozenRandomSetup,selectedAlgoId:gridSearchAlgoId,selectedAlgoLabel:gridSearchSelectedAlgo?.name||gridSearchAlgoId,estimatedMs,combos});
      setGridSearchState(prev=>({...prev,running:false,progress:1,remainingMs:0,stage:"done",simulationIndex:result.simulationCount,error:null}));
    }catch(error){
      if(gridSearchAbortRef.current===controller)gridSearchAbortRef.current=null;
      if(isAbortError(error)){
        setGridSearchState(prev=>({...prev,running:false,remainingMs:0,stage:"cancelled",error:"Gridsearch arrêté.",cancelRequested:false}));
        return;
      }
      if(gridSearchAbortRef.current===controller)gridSearchAbortRef.current=null;
      if(isAbortError(error)){
        setGridSearchState(prev=>({...prev,running:false,remainingMs:0,stage:"cancelled",error:"Gridsearch arrêté.",cancelRequested:false}));
        return;
      }
      if(isAbortError(error)){
        setGridSearchState(prev=>({...prev,running:false,remainingMs:0,stage:"cancelled",error:"Gridsearch arrêté.",cancelRequested:false}));
        if(gridSearchAbortRef.current===controller)gridSearchAbortRef.current=null;
        return;
      }
      if(isAbortError(error)){
        setMonteCarloState(prev=>({...prev,running:false,remainingMs:0,stage:"cancelled",error:"Simulation arrêtée.",cancelRequested:false}));
        if(monteCarloAbortRef.current===controller)monteCarloAbortRef.current=null;
        return;
      }
      setGridSearchState(prev=>({...prev,running:false,error:error?.message||"Le gridsearch Monte Carlo a échoué."}));
      if(monteCarloAbortRef.current===controller)monteCarloAbortRef.current=null;
      if(gridSearchAbortRef.current===controller)gridSearchAbortRef.current=null;
    }
  };

  */
  return(
    <div style={{marginBottom:14}}>
      <div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:THEME.textSecondary,marginBottom:8}}>
        <span>Min : <strong style={{color}}>{valMin}</strong></span>
        <span>Max : <strong style={{color}}>{valMax}</strong></span>
      </div>
      <div ref={trackRef} style={{position:"relative",height:28,userSelect:"none",cursor:"default"}}>
        {/* Track bg */}
        <div style={{position:"absolute",top:"50%",left:0,right:0,height:4,borderRadius:2,background:THEME.border,transform:"translateY(-50%)"}}/>
        {/* Active fill */}
        <div style={{position:"absolute",top:"50%",left:`${pctMin}%`,width:`${pctMax-pctMin}%`,height:4,borderRadius:2,background:color,transform:"translateY(-50%)"}}/>
        {/* Min thumb */}
        <div onMouseDown={e=>onMouseDown("min",e)} onTouchStart={e=>onMouseDown("min",e)}
          style={{position:"absolute",top:"50%",left:`${pctMin}%`,width:18,height:18,borderRadius:"50%",
            background:color,border:`2px solid ${THEME.appBg}`,transform:"translate(-50%,-50%)",
            cursor:"grab",zIndex:2,boxShadow:`0 0 0 3px ${color}55`,touchAction:"none"}}/>
        {/* Max thumb */}
        <div onMouseDown={e=>onMouseDown("max",e)} onTouchStart={e=>onMouseDown("max",e)}
          style={{position:"absolute",top:"50%",left:`${pctMax}%`,width:18,height:18,borderRadius:"50%",
            background:color,border:`2px solid ${THEME.appBg}`,transform:"translate(-50%,-50%)",
            cursor:"grab",zIndex:2,boxShadow:`0 0 0 3px ${color}55`,touchAction:"none"}}/>
      </div>
      {/* Ticks */}
      <div style={{display:"flex",justifyContent:"space-between",marginTop:4}}>
        {ticks.map(t=>(
          <div key={t} style={{display:"flex",flexDirection:"column",alignItems:"center",gap:1}}>
            <div style={{width:1,height:4,background:t>=valMin&&t<=valMax?color:THEME.border}}/>
            <span style={{fontSize:8,color:t>=valMin&&t<=valMax?color:THEME.textDim,fontWeight:t===valMin||t===valMax?700:400}}>{t}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── UI Helpers ───────────────────────────────────────────────────────────────
function TT({text,children}){
  const [show,setShow]=useState(false);
  return(
    <div style={{position:"relative",display:"inline-block"}} onMouseEnter={()=>setShow(true)} onMouseLeave={()=>setShow(false)}>
      {children}
      {show&&text&&<div style={{position:"absolute",bottom:"115%",left:"50%",transform:"translateX(-50%)",background:THEME.panelBgSoft,border:`1px solid ${THEME.border}`,borderRadius:8,padding:"8px 12px",fontSize:11,color:THEME.textSecondary,width:220,zIndex:200,lineHeight:1.5,pointerEvents:"none",whiteSpace:"normal",boxShadow:"0 4px 20px #0008"}}>{text}</div>}
    </div>
  );
}
function Section({title,children,titleColor,titleStyle={}}){return(<div style={{marginBottom:16}}><div style={{fontSize:9,fontWeight:700,color:titleColor||THEME.textDim,textTransform:"uppercase",letterSpacing:1,marginBottom:7,...titleStyle}}>{title}</div>{children}</div>);}
function csvDownload(rows,filename){if(!rows||!rows.length)return;const keys=Object.keys(rows[0]);const lines=[keys.join(","),...rows.map(r=>keys.map(k=>{const v=r[k];return typeof v==="string"&&v.includes(",")? `"${v}"`:v??""}).join(","))];const blob=new Blob([lines.join("\n")],{type:"text/csv"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=filename;a.click();}
function ExportBtn({onClick}){return(<button onClick={e=>{e.stopPropagation();onClick();}} title="Exporter les données" style={{background:"none",border:"1.5px solid #166534",borderRadius:6,padding:"2px 7px",fontSize:10,color:"#166534",cursor:"pointer",fontWeight:700,display:"flex",alignItems:"center",gap:3}}><span>⬇</span><span>CSV</span></button>);}
function Card({title,children,style={},onExport}){return(<div style={{background:"#a8a8a8",borderRadius:12,padding:18,border:`1px solid ${THEME.border}`,...style}}>{title&&<div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:14}}><div style={{fontWeight:700,fontSize:13,color:"#000"}}>{title}</div>{onExport&&<ExportBtn onClick={onExport}/>}</div>}{children}</div>);}

function calcMetrics(preds,rows){
  const n=rows.length;
  const mae=preds.reduce((s,p,i)=>s+Math.abs(p-rows[i].y_true),0)/n;
  const rmse=Math.sqrt(preds.reduce((s,p,i)=>s+(p-rows[i].y_true)**2,0)/n);
  const mape=preds.reduce((s,p,i)=>s+Math.abs(p-rows[i].y_true)/(Math.abs(rows[i].y_true)+1),0)/n*100;
  return{mae,rmse,mape};
}

function createMonteCarloAsyncState(){return{running:false,progress:0,elapsedMs:0,remainingMs:0,stage:"idle",simulationIndex:0,algoIndex:-1,currentAlgoId:null,error:null,cancelRequested:false};}
function createGridSearchAsyncState(){return{running:false,progress:0,elapsedMs:0,remainingMs:0,stage:"idle",simulationIndex:0,comboIndex:-1,currentComboLabel:null,error:null,cancelRequested:false};}

// ─── Phase Card (Manual mode) ─────────────────────────────────────────────────
function PhaseCard({phase,phaseIdx,expertIdx,nRows,onUpdate,onRemove,canRemove}){
  const pct=v=>Math.round(v/nRows*100);
  return(
    <div style={{background:THEME.panelBgSoft,border:`1px solid ${THEME.border}`,borderRadius:8,padding:"8px 10px",marginBottom:6}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
        <span style={{fontSize:10,fontWeight:700,color:THEME.textSecondary}}>Phase {phaseIdx+1}</span>
        {canRemove&&<button onClick={onRemove} style={{background:"transparent",border:"none",color:"#ef4444",cursor:"pointer",fontSize:13}}>×</button>}
      </div>
      <div style={{marginBottom:5}}>
        <div style={{fontSize:9,color:THEME.textMuted,marginBottom:2}}>Expert synthétique</div>
        <select value={phase.expert} onChange={e=>onUpdate({...phase,expert:e.target.value})}
          style={{width:"100%",background:THEME.panelBg,border:`1px solid ${THEME.border}`,color:THEME.textPrimary,borderRadius:5,padding:"4px 6px",fontSize:10}}>
          {ALL_SYNTHETIC.map(id=><option key={id} value={id}>{id.replace(/_/g," ")}</option>)}
        </select>
      </div>
      <div style={{marginBottom:5}}>
        <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:THEME.textMuted,marginBottom:2}}>
          <span>Longueur : {phase.end-phase.start} pts ({pct(phase.end-phase.start)}%)</span>
          <span>[{phase.start}–{phase.end}]</span>
        </div>
        <input type="range" min={phase.start+1} max={nRows-(/* leave 1 for remaining phases */0)} value={phase.end}
          onChange={e=>onUpdate({...phase,end:+e.target.value})}
          style={{width:"100%",accentColor:"#8b5cf6"}}/>
      </div>
      <div>
        <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:THEME.textMuted,marginBottom:2}}>
          <span>Bruit : {(phase.noise*100).toFixed(0)}%</span>
        </div>
        <input type="range" min={0} max={0.5} step={0.01} value={phase.noise}
          onChange={e=>onUpdate({...phase,noise:+e.target.value})}
          style={{width:"100%",accentColor:"#f472b6"}}/>
      </div>
    </div>
  );
}

// ─── Manual Expert Builder ────────────────────────────────────────────────────
function ManualExpertBuilder({rows,manualExperts,setManualExperts}){
  const n=rows.length;
  const addExpert=()=>{
    const id=`rand_expert_${manualExperts.length+1}`;
    setManualExperts(prev=>[...prev,{id,label:`M-Expert ${prev.length+1}`,phases:[{start:0,end:n,expert:ALL_SYNTHETIC[0],noise:0}]}]);
  };
  const removeExpert=i=>setManualExperts(prev=>prev.filter((_,j)=>j!==i));
  const updateExpert=(ei,updated)=>setManualExperts(prev=>prev.map((e,i)=>i===ei?updated:e));
  const addPhase=(ei)=>{
    const e=manualExperts[ei];
    const last=e.phases[e.phases.length-1];
    const mid=Math.floor((last.start+last.end)/2);
    if(mid<=last.start){return;}
    const newPhases=[...e.phases.slice(0,-1),{...last,end:mid},{start:mid,end:last.end,expert:ALL_SYNTHETIC[0],noise:0}];
    updateExpert(ei,{...e,phases:newPhases});
  };
  const removePhase=(ei,pi)=>{
    const e=manualExperts[ei];
    if(e.phases.length<=1)return;
    const newPhases=e.phases.filter((_,j)=>j!==pi);
    // Fix continuity
    for(let i=1;i<newPhases.length;i++) newPhases[i].start=newPhases[i-1].end;
    newPhases[newPhases.length-1].end=n;
    updateExpert(ei,{...e,phases:newPhases});
  };
  const updatePhase=(ei,pi,updated)=>{
    const e=manualExperts[ei];
    const newPhases=[...e.phases];
    newPhases[pi]={...updated};
    // cascade: next phase starts where this one ends
    for(let i=pi+1;i<newPhases.length;i++) newPhases[i]={...newPhases[i],start:newPhases[i-1].end};
    newPhases[newPhases.length-1]={...newPhases[newPhases.length-1],end:n};
    updateExpert(ei,{...e,phases:newPhases});
  };

  return(
    <div>
      {manualExperts.map((e,ei)=>(
        <div key={e.id} style={{background:THEME.panelBg,border:`1px solid ${THEME.border}`,borderRadius:10,padding:10,marginBottom:10}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
            <input value={e.label} onChange={ev=>updateExpert(ei,{...e,label:ev.target.value})}
              style={{background:"transparent",border:"none",color:"#60a5fa",fontWeight:700,fontSize:12,width:"60%",outline:"none"}}/>
            <button onClick={()=>removeExpert(ei)} style={{background:"#ef444422",color:"#ef4444",border:"1px solid #ef444444",borderRadius:6,padding:"2px 8px",fontSize:11,cursor:"pointer"}}>Supprimer</button>
          </div>
          {e.phases.map((ph,pi)=>(
            <PhaseCard key={pi} phase={ph} phaseIdx={pi} expertIdx={ei} nRows={n}
              onUpdate={updated=>updatePhase(ei,pi,updated)}
              onRemove={()=>removePhase(ei,pi)}
              canRemove={e.phases.length>1}/>
          ))}
          <button onClick={()=>addPhase(ei)} style={{width:"100%",background:"#8b5cf622",color:"#a78bfa",border:"1px dashed #8b5cf6",borderRadius:7,padding:"5px 0",fontSize:11,cursor:"pointer",marginTop:2}}>+ Ajouter une phase</button>
        </div>
      ))}
      {manualExperts.length<10&&(
        <button onClick={addExpert} style={{width:"100%",background:"#3b82f622",color:"#60a5fa",border:"1px dashed #3b82f6",borderRadius:8,padding:"7px 0",fontSize:11,cursor:"pointer",fontWeight:600}}>
          + Ajouter un expert manuel
        </button>
      )}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App(){
  const [rawRows,setRawRows]=useState([]);
  const [fileName,setFileName]=useState("données démo (27 lignes)");
  const fileRef=useRef();
  const [dateFrom,setDateFrom]=useState("2025-02-09T00:00");
  const [dateTo,setDateTo]=useState("2025-11-22T23:00");

  // Expert mode: "old" | "random" | "manual"
  const [expertMode,setExpertMode]=useState("old");

  // Old mode
  const [selectedExperts,setSelected]=useState(["ridge_full","rf_full","lgbm_full","elasticnet_full"]);
  // Random mode params
  const [randN,setRandN]=useState(4);
  const [randPhaseMin,setRandPhaseMin]=useState(3);
  const [randPhaseMax,setRandPhaseMax]=useState(7);
  const [randNoise,setRandNoise]=useState(0.05);
  const [generatedExperts,setGeneratedExperts]=useState([]);
  const [lastRandomSetup,setLastRandomSetup]=useState(null);
  // Manual mode
  const [manualExperts,setManualExperts]=useState([]);

  const [algoId,setAlgoId]=useState("BOA");
  const [selectedHmoeRegimes,setSelectedHmoeRegimes]=useState(()=>[...DEFAULT_HMOE_REGIME_IDS]);
  const [lossType,setLossType]=useState(DEFAULT_LOSS_TYPE);
  const [useGrad,setUseGrad]=useState(DEFAULT_USE_GRAD);
  const [ftrlP,setFtrlP]=useState(()=>({...DEFAULT_FTRL_PARAMS}));
  const [extraP,setExtraP]=useState(()=>({...DEFAULT_EXTRA_PARAMS}));
  const [showCsvInfo,setShowCsvInfo]=useState(false);
  const [showTutorial,setShowTutorial]=useState(false);
  const [prodMode,setProdMode]=useState(false);
  const [prodSelectedExperts,setProdSelectedExperts]=useState([]);
  const [results,setResults]=useState(null);
  const [running,setRunning]=useState(false);
  const [tab,setTab]=useState("forecast");
  const [horizonH,setHorizonH]=useState(48);
  const [allRuns,setAllRuns]=useState([]);
  const [visibleRuns,setVisibleRuns]=useState(new Set());
  const [cmpHorizon,setCmpHorizon]=useState(48);
  const [monteCarloCount,setMonteCarloCount]=useState(25);
  const [monteCarloAlgoIds,setMonteCarloAlgoIds]=useState(["BOA","SimpleMean"]);
  const [monteCarloState,setMonteCarloState]=useState(createMonteCarloAsyncState);
  const [monteCarloResult,setMonteCarloResult]=useState(null);
  const defaultGridSearchGroupLabel=ALGO_GROUPS[0]?.label||"";
  const defaultGridSearchAlgoId=ALGO_GROUPS[0]?.algos?.[0]?.id||"BOA";
  const gridSearchComboIdRef=useRef(1);
  const buildGridSearchCombosForAlgo=useCallback((nextAlgoId)=>{
    return getInitialGridSearchComboOverrides(nextAlgoId).map(overrides=>createGridSearchCombo(nextAlgoId,gridSearchComboIdRef.current++,overrides));
  },[]);
  const [gridSearchSimulationCount,setGridSearchSimulationCount]=useState(25);
  const [gridSearchGroupLabel,setGridSearchGroupLabel]=useState(defaultGridSearchGroupLabel);
  const [gridSearchAlgoId,setGridSearchAlgoId]=useState(defaultGridSearchAlgoId);
  const [gridSearchCombos,setGridSearchCombos]=useState(()=>buildGridSearchCombosForAlgo(defaultGridSearchAlgoId));
  const [gridSearchState,setGridSearchState]=useState(createGridSearchAsyncState);
  const [gridSearchResult,setGridSearchResult]=useState(null);
  const monteCarloAbortRef=useRef(null);
  const gridSearchAbortRef=useRef(null);

  const NON_EXPERT_COLS=useMemo(()=>new Set(["target_time","y_true","decision_time","horizon","wind_global_index","wind_norm","mom_24","mom_48","vol_24","vol_48","trend_24_gap","hour_sin","hour_cos"]),[]);
  const csvExpertCols=useMemo(()=>rawRows.length?Object.keys(rawRows[0]).filter(k=>!NON_EXPERT_COLS.has(k)):[],[rawRows,NON_EXPERT_COLS]);

  useEffect(()=>{loadRows(parseCSV(DEMO_CSV));},[]);
  useEffect(()=>()=>{monteCarloAbortRef.current?.abort();gridSearchAbortRef.current?.abort();},[]);
  useEffect(()=>{if(csvExpertCols.length)setProdSelectedExperts([...csvExpertCols]);},[csvExpertCols]);

  function loadRows(rows){
    monteCarloAbortRef.current?.abort();
    gridSearchAbortRef.current?.abort();
    const parsed=ensureHmoeFeatures(rows.map(r=>{
      const o={...r};
      Object.keys(r).forEach(k=>{
        if(k!=="decision_time"&&k!=="target_time"){const v=parseFloat(String(r[k]).replace(/\s/g,""));o[k]=isNaN(v)?0:v;}
      });
      return o;
    }));
    const normT=t=>(t||"").replace("+00:00","").replace(" ","T").slice(0,16);
    const times=parsed.map(r=>normT(r.target_time)).filter(Boolean).sort();
    if(times.length){setDateFrom(times[0]);setDateTo(times[times.length-1]);}
    setRawRows(parsed);setResults(null);setAllRuns([]);setVisibleRuns(new Set());setManualExperts([]);setGeneratedExperts([]);setLastRandomSetup(null);setMonteCarloResult(null);setMonteCarloState(createMonteCarloAsyncState());setGridSearchResult(null);setGridSearchState(createGridSearchAsyncState());
  }

  const handleFile=e=>{
    const f=e.target.files[0];if(!f)return;
    setFileName(f.name);
    const r=new FileReader();r.onload=ev=>loadRows(parseCSV(ev.target.result));r.readAsText(f);
  };

  const norm=t=>(t||"").replace("+00:00","").replace(" ","T").slice(0,16);
  const filteredRows=rawRows.filter(r=>{const t=norm(r.target_time);return(!dateFrom||t>=dateFrom)&&(!dateTo||t<=dateTo);});

  const toggleExpert=id=>setSelected(prev=>{
    if(prev.includes(id))return prev.length>2?prev.filter(e=>e!==id):prev;
    return prev.length>=10?prev:[...prev,id];
  });
  const toggleHmoeRegime=id=>setSelectedHmoeRegimes(prev=>{
    if(prev.includes(id))return prev.length>1?prev.filter(r=>r!==id):prev;
    return [...prev,id];
  });
  const toggleMonteCarloAlgo=id=>setMonteCarloAlgoIds(prev=>prev.includes(id)?prev.filter(algo=>algo!==id):[...prev,id]);

  // Build rand experts from manual config (for manual mode)
  function buildManualRandExperts(){
    if(!filteredRows.length)return[];
    return buildConfiguredExpertsData(filteredRows,manualExperts);
  }

  const lastCtxKey=useRef(null);

  const handleRun=()=>{
    let cols,augRows,randExpertsUsed=[],evalRows=filteredRows,ctxDateFrom=dateFrom,ctxDateTo=dateTo;

    if(prodMode){
      if(filteredRows.length<2)return;
      if(prodSelectedExperts.length<2)return;
      cols=prodSelectedExperts;augRows=filteredRows;
    } else if(expertMode==="old"){
      if(filteredRows.length<2)return;
      if(selectedExperts.length<2)return;
      cols=selectedExperts;augRows=filteredRows;
    } else if(expertMode==="random"){
      if(!lastRandomSetup||lastRandomSetup.rows.length<2)return;
      if(generatedExperts.length<2)return;
      randExpertsUsed=generatedExperts;
      evalRows=lastRandomSetup.rows;
      ctxDateFrom=lastRandomSetup.dateFrom;
      ctxDateTo=lastRandomSetup.dateTo;
      augRows=buildDataWithRandExperts(evalRows,randExpertsUsed);
      cols=randExpertsUsed.map(e=>e.id);
    } else {
      if(filteredRows.length<2)return;
      if(manualExperts.length<2)return;
      randExpertsUsed=buildManualRandExperts();
      augRows=buildDataWithRandExperts(filteredRows,randExpertsUsed);
      cols=randExpertsUsed.map(e=>e.id);
    }

    setRunning(true);
    setTimeout(()=>{
      const isHmoeRun=HMOE_ALGO_IDS.includes(algoId);
      const res=isHmoeRun
        ?runHmoe(augRows,cols,getHmoeBaseAlgoId(algoId),lossType,useGrad,extraP,ftrlP,selectedHmoeRegimes)
        :runAlgo(augRows,cols,algoId,lossType,useGrad,extraP,ftrlP);
      const m=calcMetrics(res.predictions,evalRows);
      const modeStr=prodMode?"prod":expertMode==="old"?"classique":expertMode==="random"?"aléatoire":"manuel";
      const label=buildAlgoRunLabel(algoId,{lossType,useGrad,extraP:{...extraP},ftrlP:{...ftrlP},selectedHmoeRegimes:[...selectedHmoeRegimes]},modeStr);
      const algoLabel=ALGOS.find(a=>a.id===algoId)?.name||algoId;
      const newRun={id:label,label,algoId,lossType,useGrad,extraP:{...extraP},ftrlP:{...ftrlP},executedAt:Date.now(),
        experts:cols,rows:evalRows,augRows,predictions:res.predictions,weightHistory:res.weightHistory,
        mae:m.mae,rmse:m.rmse,mape:m.mape,
        randExperts:randExpertsUsed,expertMode,hmoe:res.hmoe||null,selectedHmoeRegimes:[...selectedHmoeRegimes],algoLabel};
      setResults({...res,...m,rows:evalRows,augRows,experts:cols,randExperts:randExpertsUsed,expertMode,selectedHmoeRegimes:[...selectedHmoeRegimes],algoLabel,label});
      const ctxKey=JSON.stringify({dateFrom:ctxDateFrom,dateTo:ctxDateTo});
      const ctxChanged=lastCtxKey.current!==null&&lastCtxKey.current!==ctxKey;
      lastCtxKey.current=ctxKey;
      setAllRuns(prev=>{
        if(ctxChanged)return[newRun];
        const exists=prev.findIndex(r=>r.id===label);
        if(exists>=0){const next=[...prev];next[exists]=newRun;return next;}
        return[...prev,newRun];
      });
      setVisibleRuns(prev=>{if(ctxChanged)return new Set([label]);return new Set([...prev,label]);});
      setRunning(false);setTab("forecast");
    },50);
  };

  const handleGenerate=()=>{
    if(!filteredRows.length)return;
    monteCarloAbortRef.current?.abort();
    gridSearchAbortRef.current?.abort();
    const randomSetup={rows:[...filteredRows],rowCount:filteredRows.length,nExperts:randN,phaseMin:randPhaseMin,phaseMax:randPhaseMax,noiseLevel:randNoise,dateFrom,dateTo,fileName};
    const g=generateRandExperts(randomSetup.rows,randN,[randPhaseMin,randPhaseMax],randNoise);
    setLastRandomSetup(randomSetup);
    setGeneratedExperts(g);
    setMonteCarloResult(null);
    setMonteCarloState(createMonteCarloAsyncState());
    setGridSearchResult(null);
    setGridSearchState(createGridSearchAsyncState());
  };

  const exportCSV=()=>{
    if(!results)return;
    const h=["target_time","y_true","prediction",...results.experts];
    const lines=results.rows.map((r,i)=>[r.target_time,r.y_true,results.predictions[i].toFixed(3),...results.experts.map(e=>results.augRows[i][e])].join(","));
    const a=document.createElement("a");
    a.href=URL.createObjectURL(new Blob([[h.join(","),...lines].join("\n")],{type:"text/csv"}));
    a.download=`moe_${algoId}_results.csv`;a.click();
  };

  const forecastData=!results?[]:(() => {
    const rows=results.rows,n=Math.min(horizonH,rows.length),start=rows.length-n;
    return rows.slice(-n).map((r,i)=>{
      const idx=start+i,full=norm(r.target_time);
      const o={time:full,actual:r.y_true,moe:+results.predictions[idx].toFixed(1)};
      results.experts.forEach(e=>{o[e]=results.augRows[idx][e];});
      return o;
    });
  })();

  const weightData=!results?[]:results.rows.map((r,i)=>{
    const o={time:norm(r.target_time)};
    results.experts.forEach((e,k)=>{o[e]=results.weightHistory[i][k];});
    if(r.wind_global_index!==undefined) o.wind_global_index=+r.wind_global_index||0;
    return o;
  });

  const metrics=!results?null:(()=>{
    const rows=results.rows,preds=results.predictions,n=rows.length;
    const mae=preds.reduce((s,p,i)=>s+Math.abs(p-rows[i].y_true),0)/n;
    const rmse=Math.sqrt(preds.reduce((s,p,i)=>s+(p-rows[i].y_true)**2,0)/n);
    const mape=preds.reduce((s,p,i)=>s+Math.abs(p-rows[i].y_true)/(Math.abs(rows[i].y_true)+1),0)/n*100;
    const expertMetrics=results.experts.map(e=>{
      const errs=rows.map((r,i)=>results.augRows[i][e]-r.y_true);
      return{name:e,mae:(errs.reduce((s,v)=>s+Math.abs(v),0)/n).toFixed(0),rmse:Math.sqrt(errs.reduce((s,v)=>s+v**2,0)/n).toFixed(0)};
    });
    return{mae:mae.toFixed(0),rmse:rmse.toFixed(0),mape:mape.toFixed(2),n,expertMetrics};
  })();
  const hmoeSummary=!results?.hmoe?[]:results.hmoe.selectedRegimes.map(regime=>{
    const history=results.hmoe.regimeHistory.map(step=>step[regime.id]).filter(Boolean);
    const avgFirst=history.length?history.reduce((sum,step)=>sum+(step.probabilities?.[0]??0),0)/history.length:0;
    const last=history.length?history[history.length-1]:null;
    return{
      ...regime,
      avgFirst,
      avgSecond:1-avgFirst,
      lastFirst:last?.probabilities?.[0]??0,
      lastSecond:last?.probabilities?.[1]??0,
      dominant:last?.dominantBranch??0,
    };
  });

  const isOpera=OPERA_ALGO_IDS.includes(algoId);
  const isHmoe=HMOE_ALGO_IDS.includes(algoId);
  const isOperaFamily=isOpera||isHmoe;
  const curAlgo=ALGOS.find(a=>a.id===algoId);

  // Comparison
  const visibleRunsList=allRuns.filter(r=>visibleRuns.has(r.id));
  const runIndex=id=>allRuns.findIndex(r=>r.id===id);
  const cmpChartData=useMemo(()=>{
    if(!visibleRunsList.length)return[];
    const refRows=visibleRunsList[0].rows;
    const n=Math.min(cmpHorizon,refRows.length);
    const start=refRows.length-n;
    return refRows.slice(-n).map((r,i)=>{
      const idx=start+i;
      const o={time:norm(r.target_time),actual:r.y_true};
      visibleRunsList.forEach(run=>{o[`moe_${run.label}`]=+run.predictions[idx].toFixed(1);});
      return o;
    });
  },[visibleRunsList,cmpHorizon]);

  const rankings=useMemo(()=>{
    if(!visibleRunsList.length)return null;
    const byMAE=[...visibleRunsList].sort((a,b)=>a.mae-b.mae);
    const byRMSE=[...visibleRunsList].sort((a,b)=>a.rmse-b.rmse);
    const byMAPE=[...visibleRunsList].sort((a,b)=>a.mape-b.mape);
    const scoreMap={};visibleRunsList.forEach(r=>{scoreMap[r.id]={mae:0,rmse:0,mape:0};});
    byMAE.forEach((r,i)=>{scoreMap[r.id].mae=i+1;});byRMSE.forEach((r,i)=>{scoreMap[r.id].rmse=i+1;});byMAPE.forEach((r,i)=>{scoreMap[r.id].mape=i+1;});
    const general=[...visibleRunsList].sort((a,b)=>{const sa=(scoreMap[a.id].mae+scoreMap[a.id].rmse+scoreMap[a.id].mape)/3;const sb=(scoreMap[b.id].mae+scoreMap[b.id].rmse+scoreMap[b.id].mape)/3;return sa-sb;});
    return{byMAE,byRMSE,byMAPE,general,scoreMap};
  },[visibleRunsList]);

  const medalColor=rank=>rank===1?"#d39d2a":rank===2?"#8da0b8":rank===3?"#c97b43":"#60758d";

  // Mode button style
  const modeBtn=(m,label,_,color,onRed=false)=>(
    <button onClick={()=>setExpertMode(m)} style={{
      flex:1,background:expertMode===m?`${color}33`:(onRed?"rgba(255,255,255,0.12)":"transparent"),
      color:onRed?"#000000":(expertMode===m?color:THEME.textMuted),
      border:`2px solid ${expertMode===m?color:(onRed?"rgba(255,255,255,0.35)":THEME.border)}`,
      borderRadius:9,padding:"8px 4px",fontSize:10,fontWeight:expertMode===m?700:400,
      cursor:"pointer",transition:"all .15s",display:"flex",alignItems:"center",justifyContent:"center"
    }}>
      {label}
    </button>
  );

  const expertSelectionReady=prodMode?(prodSelectedExperts.length>=2):(expertMode==="old"&&selectedExperts.length>=2)||(expertMode==="random"&&generatedExperts.length>=2&&!!lastRandomSetup)||(expertMode==="manual"&&manualExperts.length>=2);
  const hasRunnableRows=(prodMode||expertMode!=="random")?(filteredRows.length>=2):(lastRandomSetup?.rows?.length||0)>=2;
  const canRun=expertSelectionReady&&hasRunnableRows&&(!isHmoe||selectedHmoeRegimes.length>=1);
  const randomConfigDirty=!!lastRandomSetup&&(randN!==lastRandomSetup.nExperts||randPhaseMin!==lastRandomSetup.phaseMin||randPhaseMax!==lastRandomSetup.phaseMax||randNoise!==lastRandomSetup.noiseLevel||dateFrom!==lastRandomSetup.dateFrom||dateTo!==lastRandomSetup.dateTo);
  const {configs:monteCarloAlgoRunConfigs,latestHmoeRun:monteCarloLatestHmoeRun}=useMemo(
    ()=>resolveMonteCarloAlgoConfigs(allRuns),
    [allRuns],
  );
  const monteCarloEstimateMs=useMemo(()=>{
    if(!lastRandomSetup||monteCarloAlgoIds.length===0||monteCarloCount<1)return 0;
    return estimateMonteCarloMs({rowCount:lastRandomSetup.rowCount,simulationCount:monteCarloCount,randomConfig:lastRandomSetup,algoIds:monteCarloAlgoIds});
  },[lastRandomSetup,monteCarloCount,monteCarloAlgoIds]);
  const monteCarloCanRun=!prodMode&&expertMode==="random"&&!!lastRandomSetup&&lastRandomSetup.rows.length>=2&&monteCarloCount>=3&&monteCarloAlgoIds.length>=2&&!monteCarloState.running;
  const monteCarloWarnings=[
    prodMode?"Le mode Prod est actif. Désactivez-le et passez en mode Aléatoire pour utiliser cette page.":null,
    !prodMode&&expertMode!=="random"?"La simulation Monte Carlo n'est disponible que si le mode Aléatoire est actif.":null,
    !lastRandomSetup?"Cliquez au moins une fois sur « Générer X experts » pour figer les conditions de génération de référence.":null,
    monteCarloCount<3?"Le nombre de simulations doit être un entier n > 2.":null,
    monteCarloAlgoIds.length<2?"Sélectionnez au moins 2 méthodes d'agrégation.":null,
  ].filter(Boolean);
  const monteCarloCurrentAlgoLabel=monteCarloState.currentAlgoId?(ALGOS.find(a=>a.id===monteCarloState.currentAlgoId)?.name||monteCarloState.currentAlgoId):null;
  const monteCarloSharedHmoeAlgoLabel=monteCarloLatestHmoeRun?(ALGOS.find(a=>a.id===monteCarloLatestHmoeRun.algoId)?.name||monteCarloLatestHmoeRun.algoId):null;
  const isAbortError=error=>error?.name==="AbortError";
  const handleStopMonteCarlo=()=>{
    if(!monteCarloAbortRef.current)return;
    monteCarloAbortRef.current.abort();
    setMonteCarloState(prev=>({...prev,stage:"cancelling",cancelRequested:true,error:null}));
  };
  const handleMonteCarloRun=async()=>{
    if(!monteCarloCanRun||!lastRandomSetup)return;
    const selectedAlgoIds=[...monteCarloAlgoIds];
    const algoRunConfigs=Object.fromEntries(
      selectedAlgoIds.map(id=>[id,cloneAlgoRunConfig(monteCarloAlgoRunConfigs[id])]),
    );
    const frozenRandomSetup=lastRandomSetup;
    const estimatedMs=monteCarloEstimateMs;
    const controller=new AbortController();
    monteCarloAbortRef.current=controller;
    setTab("montecarlo");
    setMonteCarloResult(null);
    setMonteCarloState({running:true,progress:0,elapsedMs:0,remainingMs:monteCarloEstimateMs,stage:"starting",simulationIndex:0,algoIndex:-1,currentAlgoId:null,error:null,cancelRequested:false});
    try{
      const result=await runMonteCarloSimulation({
        rows:frozenRandomSetup.rows,
        simulationCount:monteCarloCount,
        algoIds:selectedAlgoIds,
        randomConfig:frozenRandomSetup,
        algoRunConfigs,
        syntheticIds:ALL_SYNTHETIC,
        signal:controller.signal,
        onProgress:progress=>{
          setMonteCarloState(prev=>({...prev,...progress,running:progress.stage!=="done",error:null}));
        },
      });
      if(monteCarloAbortRef.current===controller)monteCarloAbortRef.current=null;
      setMonteCarloResult({...result,randomSetup:frozenRandomSetup,selectedAlgoIds,estimatedMs,algoRunConfigs});
      setMonteCarloState(prev=>({...prev,running:false,progress:1,remainingMs:0,stage:"done",simulationIndex:result.simulationCount,error:null}));
    }catch(error){
      if(monteCarloAbortRef.current===controller)monteCarloAbortRef.current=null;
      if(isAbortError(error)){
        setMonteCarloState(prev=>({...prev,running:false,remainingMs:0,stage:"cancelled",error:"Simulation arrêtée.",cancelRequested:false}));
        return;
      }
      setMonteCarloState(prev=>({...prev,running:false,error:error?.message||"La simulation Monte Carlo a échoué.",cancelRequested:false}));
    }
  };

  const gridSearchSelectedGroup=ALGO_GROUPS.find(group=>group.label===gridSearchGroupLabel)||ALGO_GROUPS[0];
  const gridSearchAvailableAlgos=gridSearchSelectedGroup?.algos||[];
  const gridSearchSelectedAlgo=ALGOS.find(algo=>algo.id===gridSearchAlgoId)||gridSearchAvailableAlgos[0]||ALGOS[0];
  const gridSearchControlSections=useMemo(
    ()=>getGridSearchControlSections(gridSearchAlgoId,LOSS_TYPES,HMOE_REGIME_TYPES),
    [gridSearchAlgoId],
  );
  const gridSearchDistinctCount=useMemo(
    ()=>new Set(gridSearchCombos.map(combo=>getGridSearchComboSignature(gridSearchAlgoId,combo))).size,
    [gridSearchAlgoId,gridSearchCombos],
  );
  const gridSearchEstimateMs=useMemo(()=>{
    if(!lastRandomSetup||!gridSearchAlgoId||gridSearchSimulationCount<1||gridSearchCombos.length===0)return 0;
    return estimateMonteCarloGridSearchMs({
      rowCount:lastRandomSetup.rowCount,
      simulationCount:gridSearchSimulationCount,
      randomConfig:lastRandomSetup,
      algoId:gridSearchAlgoId,
      comboCount:gridSearchCombos.length,
    });
  },[lastRandomSetup,gridSearchSimulationCount,gridSearchAlgoId,gridSearchCombos.length]);
  const gridSearchCanRun=expertMode==="random"
    &&!!lastRandomSetup
    &&lastRandomSetup.rows.length>=2
    &&gridSearchSimulationCount>=3
    &&gridSearchCombos.length>=2
    &&gridSearchDistinctCount>=2
    &&(!HMOE_ALGO_IDS.includes(gridSearchAlgoId)||gridSearchCombos.every(combo=>combo.selectedHmoeRegimes.length>=1))
    &&!gridSearchState.running;
  const gridSearchWarnings=[
    !prodMode&&expertMode!=="random"?"Le gridsearch Monte Carlo n'est disponible que si le mode Aléatoire est actif.":null,
    !prodMode&&!lastRandomSetup?"Cliquez au moins une fois sur « Générer X experts » pour figer les conditions de génération de référence.":null,
    gridSearchSimulationCount<3?"Le nombre de simulations doit être un entier n > 2.":null,
    gridSearchCombos.length<2?"Ajoutez au moins 2 combinaisons de paramètres.":null,
    gridSearchCombos.length>=2&&gridSearchDistinctCount<2?"Les combinaisons doivent être distinctes pour permettre le classement.":null,
    HMOE_ALGO_IDS.includes(gridSearchAlgoId)&&gridSearchCombos.some(combo=>combo.selectedHmoeRegimes.length<1)?"Chaque combinaison HMOE doit conserver au moins 1 régime.":null,
  ].filter(Boolean);
  const gridSearchCurrentLabel=gridSearchState.currentComboLabel;
  const handleStopGridSearch=()=>{
    if(!gridSearchAbortRef.current)return;
    gridSearchAbortRef.current.abort();
    setGridSearchState(prev=>({...prev,stage:"cancelling",cancelRequested:true,error:null}));
  };
  const handleGridSearchGroupChange=nextGroupLabel=>{
    const nextGroup=ALGO_GROUPS.find(group=>group.label===nextGroupLabel)||ALGO_GROUPS[0];
    const nextAlgoId=nextGroup?.algos?.[0]?.id||"BOA";
    setGridSearchGroupLabel(nextGroup.label);
    setGridSearchAlgoId(nextAlgoId);
    setGridSearchCombos(buildGridSearchCombosForAlgo(nextAlgoId));
    setGridSearchResult(null);
    setGridSearchState(createGridSearchAsyncState());
  };
  const handleGridSearchAlgoChange=nextAlgoId=>{
    setGridSearchAlgoId(nextAlgoId);
    setGridSearchCombos(buildGridSearchCombosForAlgo(nextAlgoId));
    setGridSearchResult(null);
    setGridSearchState(createGridSearchAsyncState());
  };
  const updateGridSearchCombo=(comboId,updater)=>{
    setGridSearchCombos(prev=>prev.map(combo=>combo.id===comboId?updater(combo):combo));
  };
  const updateGridSearchControl=(comboId,scope,field,value)=>{
    updateGridSearchCombo(comboId,combo=>{
      if(scope==="root")return{...combo,[field]:value};
      return{...combo,[scope]:{...combo[scope],[field]:value}};
    });
  };
  const toggleGridSearchRegime=(comboId,regimeId)=>{
    updateGridSearchCombo(comboId,combo=>{
      const selected=combo.selectedHmoeRegimes.includes(regimeId);
      const nextSelected=selected
        ?(combo.selectedHmoeRegimes.length>1?combo.selectedHmoeRegimes.filter(id=>id!==regimeId):combo.selectedHmoeRegimes)
        :[...combo.selectedHmoeRegimes,regimeId];
      return{...combo,selectedHmoeRegimes:nextSelected};
    });
  };
  const addGridSearchCombo=()=>{
    setGridSearchCombos(prev=>{
      const lastCombo=prev[prev.length-1];
      if(!lastCombo)return buildGridSearchCombosForAlgo(gridSearchAlgoId);
      const {id:_ignored,...comboSeed}=lastCombo;
      return[...prev,createGridSearchCombo(gridSearchAlgoId,gridSearchComboIdRef.current++,comboSeed)];
    });
  };
  const removeGridSearchCombo=comboId=>{
    setGridSearchCombos(prev=>prev.length>1?prev.filter(combo=>combo.id!==comboId):prev);
  };
  const handleGridSearchRun=async()=>{
    if(!gridSearchCanRun||!lastRandomSetup)return;
    const frozenRandomSetup=lastRandomSetup;
    const combos=gridSearchCombos.map((combo,index)=>({
      ...combo,
      extraP:{...combo.extraP},
      ftrlP:{...combo.ftrlP},
      selectedHmoeRegimes:[...combo.selectedHmoeRegimes],
      label:buildGridSearchComboLabel(gridSearchAlgoId,combo,index),
    }));
    const estimatedMs=gridSearchEstimateMs;
    const controller=new AbortController();
    gridSearchAbortRef.current=controller;
    setTab("gridsearchmc");
    setGridSearchResult(null);
    setGridSearchState({running:true,progress:0,elapsedMs:0,remainingMs:estimatedMs,stage:"starting",simulationIndex:0,comboIndex:-1,currentComboLabel:null,error:null,cancelRequested:false});
    try{
      const result=await runMonteCarloGridSearch({
        rows:frozenRandomSetup.rows,
        simulationCount:gridSearchSimulationCount,
        algoId:gridSearchAlgoId,
        combos,
        randomConfig:frozenRandomSetup,
        syntheticIds:ALL_SYNTHETIC,
        signal:controller.signal,
        onProgress:progress=>{
          setGridSearchState(prev=>({...prev,...progress,running:progress.stage!=="done",error:null}));
        },
      });
      if(gridSearchAbortRef.current===controller)gridSearchAbortRef.current=null;
      setGridSearchResult({...result,randomSetup:frozenRandomSetup,selectedAlgoId:gridSearchAlgoId,selectedAlgoLabel:gridSearchSelectedAlgo?.name||gridSearchAlgoId,estimatedMs,combos});
      setGridSearchState(prev=>({...prev,running:false,progress:1,remainingMs:0,stage:"done",simulationIndex:result.simulationCount,error:null}));
    }catch(error){
      setGridSearchState(prev=>({...prev,running:false,error:error?.message||"Le gridsearch Monte Carlo a échoué."}));
    }
  };

  return(
    <div style={{fontFamily:"'Inter',sans-serif",background:THEME.appBg,minHeight:"100vh",color:THEME.textPrimary,display:"flex",flexDirection:"column"}}>
      {/* Header */}
      <div style={{background:"#ffffff",borderBottom:"1px solid #d9e4f0",padding:"12px 20px",display:"flex",alignItems:"center",gap:92}}>
        <div style={{display:"flex",alignItems:"center",gap:14}}>
          <img
            src={`${process.env.PUBLIC_URL}/longlong.png`}
            alt="Air Liquide logo"
            style={{height:20,objectFit:"contain",display:"block"}}
          />
          <img
            src={`${process.env.PUBLIC_URL}/telecomlogo.png`}
            alt="Telecom logo"
            style={{height:50,objectFit:"contain",display:"block"}}
          />
        </div>
        <div>
          <div style={{fontWeight:700,fontSize:28,color:"#123ea5",fontFamily:"'Orbitron', sans-serif"}}>MoE Runner Engine - Wind Power Time Series Forecasting</div>
        </div>
        <button onClick={()=>setShowTutorial(true)} style={{marginLeft:"auto",background:"#0e2d52",color:"#fff",border:"none",borderRadius:8,padding:"6px 16px",fontSize:12,cursor:"pointer",fontWeight:600}}>? Tutorial</button>
      </div>

      <div style={{display:"flex",flex:1,overflow:"hidden",minHeight:0}}>
        {/* Sidebar */}
        <div style={{width:300,background:"#a8a8a8",borderRight:`1px solid ${THEME.border}`,padding:"14px",overflowY:"auto"}}>
          <div style={{background:"#E2001A",borderRadius:10,padding:"10px 10px 6px 10px",marginBottom:12}}>
          {/* Données */}
          <div style={{marginBottom:16}}>
            <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:7}}>
              <div style={{fontSize:13,fontWeight:800,color:"#fff",textTransform:"uppercase",letterSpacing:0.5}}>Données</div>
              <button onClick={()=>setShowCsvInfo(true)} title="Format CSV attendu" style={{background:"rgba(255,255,255,0.18)",border:"1.5px solid rgba(255,255,255,0.5)",borderRadius:"50%",width:16,height:16,display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer",padding:0,flexShrink:0}}>
                <span style={{fontSize:9,fontWeight:800,color:"#fff",lineHeight:1}}>i</span>
              </button>
              <div style={{marginLeft:"auto",display:"flex",alignItems:"center",gap:5}}>
                <span style={{fontSize:9,fontWeight:700,color:"rgba(255,255,255,0.8)",textTransform:"uppercase",letterSpacing:0.5}}>Prod</span>
                <div onClick={()=>setProdMode(v=>!v)} style={{width:30,height:16,borderRadius:8,background:prodMode?"#fff":"rgba(255,255,255,0.25)",border:"1.5px solid rgba(255,255,255,0.6)",cursor:"pointer",position:"relative",transition:"background 0.2s"}}>
                  <div style={{position:"absolute",top:1,left:prodMode?13:1,width:12,height:12,borderRadius:"50%",background:prodMode?"#E2001A":"rgba(255,255,255,0.7)",transition:"left 0.2s"}}/>
                </div>
              </div>
            </div>
            <div onClick={()=>fileRef.current.click()} style={{border:"2px dashed rgba(255,255,255,0.6)",borderRadius:8,padding:"9px",textAlign:"center",cursor:"pointer",background:"rgba(255,255,255,0.45)"}}
              onMouseEnter={e=>e.currentTarget.style.borderColor="rgba(255,255,255,0.95)"} onMouseLeave={e=>e.currentTarget.style.borderColor="rgba(255,255,255,0.6)"}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="black" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg>
              <div style={{fontSize:10,color:"#000000",fontWeight:600}}>{fileName}</div>
            </div>
            <input ref={fileRef} type="file" accept=".csv" style={{display:"none"}} onChange={handleFile}/>
          </div>
          {showCsvInfo&&(
            <div onClick={()=>setShowCsvInfo(false)} style={{position:"fixed",inset:0,background:"rgba(0,0,0,0.55)",zIndex:1000,display:"flex",alignItems:"center",justifyContent:"center"}}>
              <div onClick={e=>e.stopPropagation()} style={{background:"#fff",borderRadius:12,padding:"24px 28px",maxWidth:560,width:"90%",maxHeight:"80vh",overflowY:"auto",boxShadow:"0 8px 40px rgba(0,0,0,0.25)"}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:16}}>
                  <div style={{fontSize:14,fontWeight:800,color:"#0e2d52"}}>Format CSV attendu</div>
                  <button onClick={()=>setShowCsvInfo(false)} style={{background:"none",border:"none",fontSize:18,cursor:"pointer",color:"#666",lineHeight:1}}>×</button>
                </div>

                <div style={{fontSize:11,fontWeight:700,color:"#0e2d52",textTransform:"uppercase",letterSpacing:0.5,marginBottom:6}}>Colonnes obligatoires</div>
                <div style={{background:"#f4f7fb",borderRadius:8,padding:"10px 12px",marginBottom:12,fontSize:11,lineHeight:1.7}}>
                  <div><code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>target_time</code> - Timestamp de la valeur à prédire (ex. <em>2025-02-10 00:00:00+00:00</em>). Les timestamps doivent être continus heure par heure pour une efficacité maximale de l'engine, les trous dans la série dégradent les features de régimes. La fenêtre temporelle est automatiquement réglée sur le min et max du fichier chargé.</div>
                  <div><code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>y_true</code> - Valeur réelle observée (variable cible)</div>
                  <div><code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>expert_1, expert_2, …</code> - N colonnes de prédictions d'experts (noms libres, au moins 2)</div>
                </div>

                <div style={{fontSize:11,fontWeight:700,color:"#0e2d52",textTransform:"uppercase",letterSpacing:0.5,marginBottom:4}}>Nombre de lignes minimum</div>
                <div style={{background:"#f4f7fb",borderRadius:8,padding:"10px 12px",marginBottom:12,fontSize:11,lineHeight:1.6}}>
                  2 lignes pour faire tourner l'application. 50 lignes minimum recommandées pour que les features de régimes HMOE soient significatives (mom_48 et trend_24_gap nécessitent au moins 49-50 observations pour être calculées correctement).
                </div>

                <div style={{fontSize:11,fontWeight:700,color:"#0e2d52",textTransform:"uppercase",letterSpacing:0.5,marginBottom:6}}>Colonnes optionnelles - features de régimes HMOE</div>
                <div style={{background:"#f4f7fb",borderRadius:8,padding:"10px 12px",marginBottom:8,fontSize:11,lineHeight:1.7}}>
                  <div style={{marginBottom:6,color:"#555",lineHeight:1.5}}>Les algorithmes HMOE nécessitent des régimes de données pour fonctionner, modélisés par des features de régimes.</div>
                  <div style={{marginBottom:4,color:"#1d7a5a",fontWeight:600}}>Calculées automatiquement depuis <code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>y_true</code> et <code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>target_time</code> si absentes :</div>
                  <div><code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>hour_sin</code>, <code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>hour_cos</code> - Encodage cyclique de l'heure UTC (<em>sin/cos(2π·h/24)</em>) - régime Day/Night</div>
                  <div><code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>mom_24</code>, <code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>mom_48</code> - Momentum : différence de <code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>y_true</code> à 24 et 48 pas - régime Up/Down</div>
                  <div><code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>vol_24</code>, <code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>vol_48</code> - Écart-type glissant des variations de <code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>y_true</code> sur 24/48 pas - régime Volatility</div>
                  <div><code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>trend_24_gap</code> - Pente de régression glissante sur 24 valeurs décalées de 24 pas - régime Trend</div>
                </div>
                <div style={{background:"#fff8ed",border:"1px solid #f1d1a7",borderRadius:8,padding:"10px 12px",marginBottom:12,fontSize:11,lineHeight:1.6}}>
                  <div style={{fontWeight:700,color:"#9a5b12",marginBottom:3}}>⚠ wind_norm - cas particulier</div>
                  <div style={{color:"#7a4a0a"}}><code style={{background:"#fde8c0",borderRadius:3,padding:"1px 4px"}}>wind_norm</code> est la vitesse du vent normalisée, issue d'un fichier de données météo externe. Elle ne peut pas être reconstituée depuis les prédictions des experts seules.</div>
                  <div style={{color:"#7a4a0a",marginTop:4}}>Si absente : la colonne est mise à <strong>0</strong> pour toutes les lignes - le régime <em>Wind</em> fonctionnera mais sans signal réel. Fournir votre propre colonne <code style={{background:"#fde8c0",borderRadius:3,padding:"1px 4px"}}>wind_norm</code> pour un résultat pertinent.</div>
                </div>

                <div style={{fontSize:11,fontWeight:700,color:"#0e2d52",textTransform:"uppercase",letterSpacing:0.5,marginBottom:6}}>Colonnes ignorées</div>
                <div style={{background:"#f4f7fb",borderRadius:8,padding:"10px 12px",marginBottom:12,fontSize:11,lineHeight:1.6,color:"#555"}}>
                  <code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>decision_time</code>, <code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>horizon</code>, <code style={{background:"#e0e7f0",borderRadius:3,padding:"1px 4px"}}>wind_global_index</code> (index brut du vent pour le plotting, issu d'un fichier prior externe), et toute autre colonne non reconnue sont ignorées silencieusement.
                </div>

                <div style={{fontSize:11,fontWeight:700,color:"#0e2d52",textTransform:"uppercase",letterSpacing:0.5,marginBottom:6}}>Exemple minimal</div>
                <div style={{background:"#0e2d52",borderRadius:8,padding:"10px 12px",fontSize:10,lineHeight:1.7,overflowX:"auto"}}>
                  <div style={{color:"#7ecfea",fontFamily:"monospace",whiteSpace:"pre"}}>{"target_time,y_true,expert_1,expert_2"}</div>
                  <div style={{color:"#c8e6c9",fontFamily:"monospace",whiteSpace:"pre"}}>{"2025-01-01 00:00:00+00:00,850.0,820.5,910.2"}</div>
                  <div style={{color:"#c8e6c9",fontFamily:"monospace",whiteSpace:"pre"}}>{"2025-01-01 01:00:00+00:00,900.0,880.1,920.7"}</div>
                </div>
              </div>
            </div>
          )}

          {showTutorial&&(
            <div onClick={()=>setShowTutorial(false)} style={{position:"fixed",inset:0,background:"rgba(0,0,0,0.6)",zIndex:1000,display:"flex",alignItems:"center",justifyContent:"center"}}>
              <div onClick={e=>e.stopPropagation()} style={{background:"#fff",borderRadius:14,padding:"28px 32px",maxWidth:640,width:"92%",maxHeight:"85vh",overflowY:"auto",boxShadow:"0 8px 40px rgba(0,0,0,0.3)"}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:20}}>
                  <div style={{fontSize:15,fontWeight:800,color:"#0e2d52"}}>Guide d'utilisation</div>
                  <button onClick={()=>setShowTutorial(false)} style={{background:"none",border:"none",fontSize:20,cursor:"pointer",color:"#666",lineHeight:1}}>×</button>
                </div>

                {/* Sub-box: Panneau latéral de paramètres */}
                <div style={{background:"#f4f7fb",border:"1px solid #d0dae8",borderRadius:10,padding:"14px 16px",marginBottom:16}}>
                  <div style={{fontSize:11,fontWeight:800,color:"#0e2d52",textTransform:"uppercase",letterSpacing:0.5,marginBottom:12}}>Panneau latéral de paramètres</div>
                  {[
                    {num:"1",title:"Charger vos données",color:"#E2001A",text:"Chargez un fichier CSV via la zone « Données » (panneau gauche). Le format attendu et les détails sont précisés dans le bouton « i » à côté de « Données ». Activez le toggle Prod si vos données sont réelles avec vos propres experts nommés. En mode normal, choisissez Aléatoire, Manuel ou Classique pour générer ou sélectionner des experts."},
                    {num:"2",title:"Configurer la fenêtre temporelle",color:"#E2001A",text:"La fenêtre De/À se règle automatiquement sur l'étendue du CSV chargé. Vous pouvez la restreindre pour ne travailler que sur une sous-période."},
                    {num:"3",title:"Choisir un algorithme",color:"#0e2d52",text:"Sélectionnez l'algorithme d'agrégation dans la box bleue. MOE BOA/MLpol/MLprod/FTRL sont les méthodes online opera. Les HMOE ajoutent un routage par régimes de données. Les méthodes classiques (Ridge, InvMSE...) servent de baseline. Cliquez sur Run."},
                  ].map(({num,title,color,text})=>(
                    <div key={num} style={{display:"flex",gap:12,marginBottom:12}}>
                      <div style={{width:22,height:22,borderRadius:"50%",background:color,color:"#fff",fontSize:11,fontWeight:800,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,marginTop:1}}>{num}</div>
                      <div>
                        <div style={{fontSize:12,fontWeight:700,color,marginBottom:3}}>{title}</div>
                        <div style={{fontSize:11,color:"#444",lineHeight:1.55}}>{text}</div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Sub-box: Pages */}
                <div style={{background:"#f4f7fb",border:"1px solid #d0dae8",borderRadius:10,padding:"14px 16px",marginBottom:16}}>
                  <div style={{fontSize:11,fontWeight:800,color:"#0e2d52",textTransform:"uppercase",letterSpacing:0.5,marginBottom:12}}>Pages</div>
                  {[
                    {num:"4",title:"Page Prévisions",color:"#7f64c3",text:"Visualise la prévision MOE contre les valeurs réelles sur l'horizon choisi (24h, 48h, 72h). Les métriques MAE, RMSE et MAPE sont affichées par expert et pour le MOE global."},
                    {num:"5",title:"Page Poids dynamiques",color:"#c69427",text:"Montre l'évolution des poids attribués à chaque expert au fil du temps. Utile pour comprendre quels experts dominent selon les régimes."},
                    {num:"6",title:"Page Comparaison simple",color:"#2f8d73",text:"Lancez plusieurs runs avec des algorithmes ou paramètres différents et comparez leurs métriques côte à côte. Chaque run est sauvegardé et affiché dans le graphique multi-courbes."},
                    {num:"7",title:"Page MC Simulation",color:"#4c72b8",text:"Disponible en mode Aléatoire uniquement. Lance N simulations Monte Carlo sur les mêmes conditions figées, puis calcule les moyennes n-avg de MAE, RMSE, MAPE pour comparer plusieurs méthodes de façon robuste."},
                    {num:"8",title:"Page MC Gridsearch",color:"#b42318",text:"Disponible en mode Aléatoire uniquement. Compare une seule méthode contre elle-même avec plusieurs jeux de paramètres, via N simulations Monte Carlo. Permet d'identifier le meilleur réglage de façon statistiquement significative."},
                  ].map(({num,title,color,text})=>(
                    <div key={num} style={{display:"flex",gap:12,marginBottom:12}}>
                      <div style={{width:22,height:22,borderRadius:"50%",background:color,color:"#fff",fontSize:11,fontWeight:800,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,marginTop:1}}>{num}</div>
                      <div>
                        <div style={{fontSize:12,fontWeight:700,color,marginBottom:3}}>{title}</div>
                        <div style={{fontSize:11,color:"#444",lineHeight:1.55}}>{text}</div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Export CSV — standalone, no box */}
                <div style={{display:"flex",gap:12}}>
                  <div style={{width:22,height:22,borderRadius:"50%",background:"#166534",color:"#fff",fontSize:11,fontWeight:800,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,marginTop:1}}>★</div>
                  <div>
                    <div style={{fontSize:12,fontWeight:700,color:"#166534",marginBottom:3}}>Export CSV</div>
                    <div style={{fontSize:11,color:"#444",lineHeight:1.55}}>Tous les graphiques et tableaux disposent d'un bouton ⬇ CSV en haut à droite. Les données exportées correspondent uniquement au run affiché au moment du clic. Un nouveau run écrase les données précédentes, pensez à exporter avant de relancer.</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Fenêtre temporelle */}
          <Section title="Fenêtre temporelle" titleColor="#fff" titleStyle={{fontSize:13,fontWeight:800,letterSpacing:0.5}}>
            <div style={{background:"rgba(255,255,255,0.38)",borderRadius:8,padding:"8px 10px"}}>
            {[["De",dateFrom,setDateFrom,"2025-02-09T00:00",dateTo],["À",dateTo,setDateTo,dateFrom,"2025-11-22T23:00"]].map(([lbl,val,set,mn,mx])=>(
              <div key={lbl} style={{marginBottom:6}}>
                <div style={{fontSize:10,color:"#fff",marginBottom:3}}>{lbl}</div>
                <input type="datetime-local" value={val} min={mn} max={mx} onChange={e=>set(e.target.value)}
                  style={{width:"100%",background:"rgba(255,255,255,0.25)",border:"1px solid rgba(255,255,255,0.5)",color:"#000000",borderRadius:6,padding:"5px 7px",fontSize:11,boxSizing:"border-box"}}/>
              </div>
            ))}
            <div style={{fontSize:10,color:"#fff"}}>{filteredRows.length} lignes dans la fenêtre</div>
            </div>
          </Section>

          {/* Expert mode selector */}
          {!prodMode&&(<div style={{marginBottom:0}}>
            <div style={{fontSize:13,fontWeight:800,color:"#fff",textTransform:"uppercase",letterSpacing:0.5,marginBottom:7}}>Mode de sélection des experts</div>
            <div style={{display:"flex",gap:5,background:"rgba(255,255,255,0.38)",borderRadius:8,padding:"6px 8px"}}>
              {modeBtn("random","Aléatoire",null,"#d4b0ff",true)}
              {modeBtn("manual","Manuel",null,"#6eedc0",true)}
              {modeBtn("old","Classique",null,"#93caff",true)}
            </div>
          </div>)}
          </div>{/* fin zone rouge */}

          <div style={{marginBottom:12,background:"#ffffff",borderRadius:10,padding:"10px 10px 6px 10px"}}>
            {/* ── PROD MODE ── */}
            {prodMode&&(
              <div>
                <div style={{fontSize:9,color:"#000",fontWeight:800,textTransform:"uppercase",marginBottom:6}}>Experts du CSV ({prodSelectedExperts.length}/{csvExpertCols.length} sélectionnés)</div>
                {csvExpertCols.length===0&&<div style={{fontSize:10,color:THEME.textMuted}}>Chargez un CSV pour voir les experts.</div>}
                <div style={{display:"flex",flexWrap:"wrap",gap:3,marginBottom:4}}>
                  {csvExpertCols.map((col,i)=>{
                    const sel=prodSelectedExperts.includes(col);
                    const idx=prodSelectedExperts.indexOf(col);
                    return(
                      <button key={col} onClick={()=>setProdSelectedExperts(prev=>sel?(prev.length>2?prev.filter(c=>c!==col):prev):[...prev,col])}
                        style={{background:sel?PALETTE[idx%PALETTE.length]+"33":"#ddeaf8",color:sel?PALETTE[idx%PALETTE.length]:"#000",border:`1px solid ${sel?PALETTE[idx%PALETTE.length]:"#b8d0ec"}`,borderRadius:5,padding:"3px 6px",fontSize:9.5,cursor:"pointer"}}>
                        {col.replace(/_/g," ")}
                      </button>
                    );
                  })}
                </div>
                {prodSelectedExperts.length<2&&<div style={{fontSize:10,color:"#ef4444",marginTop:4}}>⚠ Sélectionnez au moins 2 experts</div>}
              </div>
            )}
            {/* ── RANDOM MODE ── */}
            {!prodMode&&expertMode==="random"&&(
              <div>
                <div style={{display:"flex",justifyContent:"space-between",marginBottom:2}}>
                  <span style={{fontSize:10,color:"#000",fontWeight:800,textTransform:"uppercase"}}>Nb d'experts (rand_N)</span>
                  <span style={{fontSize:10,color:"#000",fontWeight:700}}>{randN}</span>
                </div>
                <input type="range" min={2} max={10} step={1} value={randN} onChange={e=>setRandN(+e.target.value)} style={{width:"100%",accentColor:"#0e2d52",marginBottom:10}}/>

                <div style={{fontSize:10,color:"#000",fontWeight:800,textTransform:"uppercase",marginBottom:8}}>Range du nb de phases random par expert</div>
                <DoubleSlider min={2} max={10} valMin={randPhaseMin} valMax={randPhaseMax} onChangeMin={setRandPhaseMin} onChangeMax={setRandPhaseMax} color="#0e2d52"/>

                <div style={{display:"flex",justifyContent:"space-between",marginBottom:2,marginTop:8}}>
                  <span style={{fontSize:10,color:"#000",fontWeight:800,textTransform:"uppercase"}}>Niveau de bruit</span>
                  <span style={{fontSize:10,color:"#000",fontWeight:700}}>{(randNoise*100).toFixed(0)}%</span>
                </div>
                <input type="range" min={0} max={0.5} step={0.01} value={randNoise} onChange={e=>setRandNoise(+e.target.value)} style={{width:"100%",accentColor:"#E2001A",marginBottom:12}}/>

                <button onClick={handleGenerate} disabled={!filteredRows.length}
                  style={{width:"100%",background:"#fff0f0",color:"#E2001A",border:"2px solid #8B0000",borderRadius:8,padding:"8px 0",fontSize:12,fontWeight:700,cursor:"pointer",marginBottom:8}}>
                  ▶ Générer {randN} experts
                </button>

                {generatedExperts.length>0&&(
                  <div>
                    <div style={{fontSize:9,color:"#000",marginBottom:6,fontWeight:800,textTransform:"uppercase"}}>Experts générés</div>
                    {generatedExperts.map((e,i)=>(
                      <div key={e.id} style={{background:THEME.panelBgSoft,border:`1px solid ${THEME.border}`,borderRadius:8,padding:"8px 10px",marginBottom:5}}>
                        <div style={{fontSize:11,fontWeight:700,color:PALETTE[i%PALETTE.length],marginBottom:4}}>{e.label}</div>
                        <div style={{fontSize:9,color:"#000",marginBottom:2}}>{e.phases.length} phases · bruit {(e.noiseLevel*100).toFixed(0)}%</div>
                        {e.phases.map((ph,pi)=>(
                          <div key={pi} style={{fontSize:9,color:"#000",padding:"2px 0",borderTop:pi>0?`1px solid ${THEME.border}`:"none"}}>
                            <span style={{color:"#000"}}>Ph{pi+1} [{ph.start}–{ph.end}] </span>
                            {ph.expert.replace(/_/g," ")}
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                )}
                {generatedExperts.length<2&&<div style={{fontSize:10,color:"#ef4444",marginTop:4}}>⚠ Générez au moins 2 experts</div>}
              </div>
            )}

            {/* ── MANUAL MODE ── */}
            {!prodMode&&expertMode==="manual"&&(
              <div>
                {filteredRows.length>0
                  ?<ManualExpertBuilder rows={filteredRows} manualExperts={manualExperts} setManualExperts={setManualExperts}/>
                  :<div style={{fontSize:10,color:THEME.textMuted}}>Chargez des données d'abord.</div>
                }
                {manualExperts.length<2&&<div style={{fontSize:10,color:"#ef4444",marginTop:6}}>⚠ Créez au moins 2 experts</div>}
              </div>
            )}

            {/* ── OLD MODE ── */}
            {!prodMode&&expertMode==="old"&&(
              <div>
                <div style={{fontSize:9,color:"#000",marginBottom:6,fontStyle:"italic"}}>{selectedExperts.length}/10 experts sélectionnés</div>
                {EXPERT_GROUPS.map(g=>(
                  <div key={g.id} style={{marginBottom:9}}>
                    <div style={{fontSize:9,color:g.id==="bloc0"?"#0e2d52":g.color,fontWeight:800,marginBottom:4,textTransform:"uppercase",letterSpacing:0.5}}>
                      {g.id==="bloc0"?"Benchmarks":g.label}
                    </div>
                    <div style={{display:"flex",flexWrap:"wrap",gap:3}}>
                      {g.experts.map(exp=>{
                        const sel=selectedExperts.includes(exp.id),idx=selectedExperts.indexOf(exp.id);
                        return(
                          <TT key={exp.id} text={exp.desc}>
                            <button onClick={()=>toggleExpert(exp.id)} style={{
                              background:sel?PALETTE[idx%PALETTE.length]+"33":"#ddeaf8",
                              color:sel?PALETTE[idx%PALETTE.length]:"#000",
                              border:`1px solid ${sel?PALETTE[idx%PALETTE.length]:"#b8d0ec"}`,
                              borderRadius:5,padding:"3px 6px",fontSize:9.5,cursor:"pointer"
                            }}>{exp.id.replace(/_/g," ")}</button>
                          </TT>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div style={{background:"#003C8F",borderRadius:10,padding:"10px 10px 6px 10px",marginBottom:12}}>
          {/* Algorithme */}
          <Section title="Algorithme d'agrégation" titleColor="#fff" titleStyle={{fontSize:13,fontWeight:800,letterSpacing:0.5}}>
            {ALGO_GROUPS.map((g,gi)=>{
              const boxBg="#1e5fcc";
              return(
              <div key={g.label} style={{marginBottom:8,background:boxBg,borderRadius:8,padding:"8px 10px"}}>
                <div style={{fontSize:9,color:"#fff",fontWeight:800,textTransform:"uppercase",letterSpacing:0.5,marginBottom:6}}>{g.label}</div>
                <div style={{display:"flex",flexDirection:"column",gap:3}}>
                  {g.algos.map(a=>(
                    <TT key={a.id} text={a.desc}>
                      <button onClick={()=>setAlgoId(a.id)} style={{
                        width:"100%",textAlign:"left",
                        background:algoId===a.id?"#ffffff":"#6898e8",
                        color:"#000",
                        border:`1px solid ${algoId===a.id?"#7aaaf0":"#4a78d8"}`,
                        borderRadius:7,padding:"5px 10px",fontSize:11,cursor:"pointer",fontWeight:algoId===a.id?700:400
                      }}>{a.name}</button>
                    </TT>
                  ))}
                </div>
              </div>
            );})}
            {isOperaFamily&&(
              <>
                <div style={{fontSize:10,color:"#fff",marginBottom:3,marginTop:6}}>Loss function</div>
                <select value={lossType} onChange={e=>setLossType(e.target.value)}
                  style={{width:"100%",background:"rgba(255,255,255,0.15)",border:"1px solid rgba(255,255,255,0.4)",color:"#fff",borderRadius:6,padding:"5px 7px",fontSize:11,marginBottom:8}}>
                  {LOSS_TYPES.map(l=><option key={l.id} value={l.id} style={{color:"#000",background:"#fff"}}>{l.label}</option>)}
                </select>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
                  <span style={{fontSize:11,color:"#fff"}}>Gradient mode</span>
                  <div onClick={()=>setUseGrad(v=>!v)} style={{width:36,height:20,borderRadius:10,background:useGrad?"#4c72b8":"rgba(255,255,255,0.3)",position:"relative",cursor:"pointer",transition:"background .2s"}}>
                    <div style={{position:"absolute",top:3,left:useGrad?18:3,width:14,height:14,borderRadius:"50%",background:"#fff",transition:"left .2s"}}/>
                  </div>
                </div>
              </>
            )}
            {isHmoe&&(
              <div style={{borderTop:"1px solid rgba(255,255,255,0.25)",paddingTop:8,marginTop:4}}>
                <div style={{fontSize:10,color:"#fff",marginBottom:6}}>Regimes HMOE actifs ({selectedHmoeRegimes.length}/{HMOE_REGIME_TYPES.length})</div>
                <div style={{display:"flex",flexDirection:"column",gap:5}}>
                  {HMOE_REGIME_TYPES.map(regime=>{
                    const selected=selectedHmoeRegimes.includes(regime.id);
                    return(
                      <button key={regime.id} onClick={()=>toggleHmoeRegime(regime.id)} style={{
                        width:"100%",textAlign:"left",background:selected?"rgba(255,255,255,0.22)":"rgba(255,255,255,0.07)",
                        color:"#fff",border:`1px solid ${selected?"rgba(255,255,255,0.8)":"rgba(255,255,255,0.25)"}`,
                        borderRadius:8,padding:"7px 10px",cursor:"pointer"
                      }}>
                        <div style={{fontSize:11,fontWeight:700}}>{regime.label}</div>
                        <div style={{fontSize:9,color:"rgba(255,255,255,0.75)",marginTop:2}}>{regime.describeFeatures}</div>
                      </button>
                    );
                  })}
                </div>
                {selectedHmoeRegimes.length<1&&<div style={{fontSize:10,color:"#ffaaaa",marginTop:6}}>Selectionnez au moins 1 regime HMOE.</div>}
              </div>
            )}
            {curAlgo&&curAlgo.params.length>0&&(
              <div style={{borderTop:"1px solid rgba(255,255,255,0.25)",paddingTop:8,marginTop:4}}>
                {curAlgo.params.map(p=>(
                  <div key={p.id} style={{marginBottom:8}}>
                    <div style={{display:"flex",justifyContent:"space-between",marginBottom:2}}>
                      <span style={{fontSize:10,color:"#fff"}}>{p.label}</span>
                      {p.type==="slider"&&<span style={{fontSize:10,color:"#fff",fontWeight:600}}>{getHmoeBaseAlgoId(algoId)==="FTRL"?(ftrlP[p.id]??p.default):(extraP[p.id]??p.default)}</span>}
                    </div>
                    {p.type==="slider"&&<input type="range" min={p.min} max={p.max} step={p.step}
                      value={getHmoeBaseAlgoId(algoId)==="FTRL"?(ftrlP[p.id]??p.default):(extraP[p.id]??p.default)}
                      onChange={e=>getHmoeBaseAlgoId(algoId)==="FTRL"?setFtrlP(prev=>({...prev,[p.id]:+e.target.value})):setExtraP(prev=>({...prev,[p.id]:+e.target.value}))}
                      style={{width:"100%",accentColor:"#a8c8f8"}}/>}
                    {p.type==="select"&&<select value={getHmoeBaseAlgoId(algoId)==="FTRL"?(ftrlP[p.id]??p.default):(extraP[p.id]??p.default)}
                      onChange={e=>getHmoeBaseAlgoId(algoId)==="FTRL"?setFtrlP(prev=>({...prev,[p.id]:parseFloat(e.target.value)})):setExtraP(prev=>({...prev,[p.id]:parseFloat(e.target.value)}))}
                      style={{background:"rgba(255,255,255,0.15)",border:"1px solid rgba(255,255,255,0.4)",color:"#fff",borderRadius:5,padding:"3px 6px",fontSize:10,width:"100%"}}>
                      {p.options.map(v=><option key={v} value={v} style={{color:"#000",background:"#fff"}}>{v}</option>)}
                    </select>}
                  </div>
                ))}
              </div>
            )}
          </Section>
          </div>{/* fin zone bleue */}

          <button onClick={handleRun} disabled={running||!hasRunnableRows||!canRun}
            style={{background:running||!canRun?THEME.border:"#E2001A",color:"#fff",border:"none",borderRadius:10,padding:"11px 0",fontWeight:700,fontSize:13,cursor:running||!canRun?"not-allowed":"pointer",width:"100%"}}>
            {running?"⏳ Calcul…":"▶  Run"}
          </button>
        </div>

        {/* Main */}
        <div style={{flex:1,overflowY:"auto",padding:18,display:"flex",flexDirection:"column",gap:14}}>
          {/* Tabs */}
          <div style={{display:"flex",alignItems:"center",gap:8,flexWrap:"wrap"}}>
            {[["forecast","Prévisions"],["weights","Poids dynamiques"],["compare","Comparaison simple des méthodes d'aggrégation"]].map(([t,lbl])=>(
              <button key={t} onClick={()=>setTab(t)} style={{
                background:tab===t?"#E2001A":"#fff",color:tab===t?"#fff":"#0e2d52",
                border:`1px solid ${tab===t?"#E2001A":"#b0c4d8"}`,borderRadius:8,
                padding:"7px 16px",fontSize:12,fontWeight:600,cursor:"pointer"
              }}>
                {lbl}
                {t==="compare"&&allRuns.length>0&&<span style={{marginLeft:5,background:"#0e2d52",color:"#fff",borderRadius:"50%",width:15,height:15,fontSize:8,fontWeight:700,display:"inline-flex",alignItems:"center",justifyContent:"center",verticalAlign:"middle"}}>{allRuns.length}</span>}
              </button>
            ))}
            <button onClick={()=>setTab("montecarlo")} style={{
              background:tab==="montecarlo"?"#E2001A":"#fff",color:tab==="montecarlo"?"#fff":"#0e2d52",
              border:`1px solid ${tab==="montecarlo"?"#E2001A":"#b0c4d8"}`,borderRadius:8,
              padding:"7px 16px",fontSize:12,fontWeight:600,cursor:"pointer"
            }}>
              Comparaison par simulation de Monte Carlo
            </button>
            <button onClick={()=>setTab("gridsearchmc")} style={{
              background:tab==="gridsearchmc"?"#E2001A":"#fff",color:tab==="gridsearchmc"?"#fff":"#0e2d52",
              border:`1px solid ${tab==="gridsearchmc"?"#E2001A":"#b0c4d8"}`,borderRadius:8,
              padding:"7px 16px",fontSize:12,fontWeight:600,cursor:"pointer"
            }}>
              Monte Carlo Gridsearch par méthode
            </button>
            {tab==="forecast"&&(
              <div style={{marginLeft:"auto",display:"flex",gap:4}}>
                {[24,48,72].map(h=>(
                  <button key={h} onClick={()=>setHorizonH(h)} style={{
                    background:horizonH===h?"#E2001A":"#fff",color:horizonH===h?"#fff":"#0e2d52",
                    border:`1px solid ${horizonH===h?"#E2001A":"#b0c4d8"}`,borderRadius:7,padding:"5px 12px",fontSize:12,fontWeight:600,cursor:"pointer"
                  }}>{h}h</button>
                ))}
              </div>
            )}
          </div>

          {/* ── FORECAST ── */}
          {tab==="forecast"&&(
            <>
              {!results&&<div style={{textAlign:"center",color:"#dbe7ff",padding:80,fontSize:13}}>Sélectionnez votre mode, configurez vos experts et cliquez sur <strong style={{color:"#ffffff"}}>Exécuter</strong></div>}
              {results&&(
                <>
                  <Card title={`Prévisions vs Réel - ${horizonH}h : ${results.label}`} style={{background:"#a8a8a8"}} onExport={()=>csvDownload(forecastData,`previsions_${results.label}.csv`)}>
                    <div style={{background:"#fff",borderRadius:8,padding:"8px 8px 4px 8px"}}>
                    <ResponsiveContainer width="100%" height={310}>
                      <LineChart data={forecastData} margin={{top:4,right:10,left:0,bottom:0}}>
                        <CartesianGrid strokeDasharray="3 3" stroke={THEME.grid}/>
                        <XAxis dataKey="time" stroke={THEME.textMuted} height={36} interval={horizonH===24?1:horizonH===48?3:7} tickFormatter={v=>norm(v).slice(11,13)+"h"} tick={{fontSize:10,fill:THEME.textSecondary}}/>
                        <YAxis stroke={THEME.textMuted} tick={{fontSize:9}} unit=" MW" width={58}/>
                        <Tooltip contentStyle={{background:"#e8e8e8",border:"1px solid #c8c8c8",borderRadius:8,fontSize:10,color:"#0e2d52"}}/>
                        <Legend wrapperStyle={{fontSize:10}}/>
                        {results.experts.map((e,i)=>(
                          <Line key={e} type="monotone" dataKey={e} dot={false} stroke={PALETTE[i%PALETTE.length]} strokeWidth={1.2} strokeOpacity={0.55} name={e.replace(/rand_expert_/,"R-Exp ").replace(/_/g," ")}/>
                        ))}
                        <Line type="monotone" dataKey="actual" dot={false} stroke="#173c66" strokeWidth={2.2} strokeDasharray="5 3" name="y_true"/>
                        <Line type="monotone" dataKey="moe" dot={false} stroke="#c15a86" strokeWidth={2.5} name={`MoE ${algoId}`}/>
                      </LineChart>
                    </ResponsiveContainer>
                    </div>
                  </Card>

                  {/* Rand expert description card */}
                  {results.randExperts&&results.randExperts.length>0&&(
                    <Card title="Structure des experts générés">
                      <div style={{background:"#fff",borderRadius:8,padding:"8px"}}>
                      <div style={{display:"flex",flexWrap:"wrap",gap:10}}>
                        {results.randExperts.map((e,i)=>(
                          <div key={e.id} style={{background:THEME.panelBgSoft,border:`1px solid ${PALETTE[i%PALETTE.length]}66`,borderRadius:10,padding:"10px 12px",minWidth:180,flex:"1 1 180px"}}>
                            <div style={{fontSize:11,fontWeight:700,color:PALETTE[i%PALETTE.length],marginBottom:6}}>{e.label}</div>
                            <div style={{fontSize:9,color:"#000",marginBottom:5}}>{e.phases.length} phases · bruit {results.expertMode==="random"?(e.noiseLevel*100).toFixed(0):"-"}%</div>
                            {e.phases.map((ph,pi)=>(
                              <div key={pi} style={{display:"flex",alignItems:"center",gap:4,marginBottom:3}}>
                                <div style={{width:`${Math.round((ph.end-ph.start)/results.rows.length*100)}%`,minWidth:4,height:6,borderRadius:3,background:PALETTE[(i+pi+1)%PALETTE.length]}}/>
                                <span style={{fontSize:8,color:"#000"}}>{ph.expert.replace(/_/g," ")}</span>
                              </div>
                            ))}
                          </div>
                        ))}
                      </div>
                      </div>
                    </Card>
                  )}

                  {metrics&&(
                    <div style={{display:"flex",gap:10}}>
                      {[{label:"MAE (MoE)",val:`${metrics.mae} MW`,c:"#4c72b8"},{label:"RMSE (MoE)",val:`${metrics.rmse} MW`,c:"#2f8d73"},{label:"MAPE (MoE)",val:`${metrics.mape}%`,c:"#c69427"},{label:"N points",val:metrics.n,c:"#7f64c3"}].map(m=>(
                        <div key={m.label} style={{flex:1,background:"#a8a8a8",borderRadius:10,padding:"10px 14px"}}>
                          <div style={{fontSize:10,color:"#000",fontWeight:600,marginBottom:6}}>{m.label}</div>
                          <div style={{background:"#fff",borderRadius:6,padding:"6px 10px",display:"inline-block",minWidth:"100%",boxSizing:"border-box"}}>
                            <span style={{fontSize:18,fontWeight:700,color:m.c}}>{m.val}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {results.hmoe&&hmoeSummary.length>0&&(
                    <Card title="Regimes HMOE actifs">
                      <div style={{background:"#fff",borderRadius:8,padding:"8px"}}>
                      <div style={{display:"flex",flexWrap:"wrap",gap:10}}>
                        {hmoeSummary.map((regime,i)=>(
                          <div key={regime.id} style={{background:"#fff",border:`1px solid ${PALETTE[i%PALETTE.length]}66`,borderRadius:10,padding:"10px 12px",minWidth:190,flex:"1 1 190px"}}>
                            <div style={{fontSize:11,fontWeight:700,color:PALETTE[i%PALETTE.length],marginBottom:4}}>{regime.label}</div>
                            <div style={{fontSize:9,color:THEME.textMuted,marginBottom:8,fontStyle:"italic"}}>{regime.describeFeatures}</div>
                            <div style={{fontSize:10,color:THEME.textSecondary,fontWeight:700}}>{regime.components[0]} avg {(regime.avgFirst*100).toFixed(1)}%</div>
                            <div style={{fontSize:10,color:THEME.textSecondary,fontWeight:700}}>{regime.components[1]} avg {(regime.avgSecond*100).toFixed(1)}%</div>
                            <div style={{fontSize:9,color:THEME.textSecondary,marginTop:8}}>Dernier pas: {regime.components[0]} {(regime.lastFirst*100).toFixed(1)}% · {regime.components[1]} {(regime.lastSecond*100).toFixed(1)}%</div>
                            <div style={{fontSize:9,color:"#1d7a5a",marginTop:4,fontWeight:700}}>Dominant (dernier pas) : {regime.components[regime.dominant]||regime.components[0]}</div>
                          </div>
                        ))}
                      </div>
                      </div>
                    </Card>
                  )}

                  {metrics&&(
                    <Card title="Erreurs par expert vs MoE" onExport={()=>csvDownload([{expert:"MoE",mae:metrics.mae,rmse:metrics.rmse,mape:metrics.mape},...metrics.expertMetrics.map(e=>({expert:e.name,mae:e.mae,rmse:e.rmse,mape:""}))],`erreurs_experts_${results.label}.csv`)}>
                      <div style={{background:"#fff",borderRadius:8,padding:"8px"}}>
                      <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                        <thead>
                          <tr style={{borderBottom:`1px solid ${THEME.border}`}}>
                            {["Expert","MAE (MW)","RMSE (MW)"].map(h=><th key={h} style={{textAlign:"left",padding:"6px 10px",color:THEME.textMuted,fontWeight:600}}>{h}</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          <tr style={{background:"#dde7f4",borderBottom:`1px solid ${THEME.border}`}}>
                            <td style={{padding:"8px 10px",fontWeight:700,color:"#c15a86"}}>MoE {algoId}</td>
                            <td style={{padding:"8px 10px",color:"#1d7a5a",fontWeight:700}}>{metrics.mae}</td>
                            <td style={{padding:"8px 10px",color:"#1d7a5a",fontWeight:700}}>{metrics.rmse}</td>
                          </tr>
                          {metrics.expertMetrics.map((e,i)=>(
                            <tr key={e.name} style={{borderBottom:`1px solid ${THEME.border}`}}>
                              <td style={{padding:"6px 10px",color:PALETTE[i%PALETTE.length]}}>{e.name.replace(/rand_expert_/,"R-Exp ").replace(/_/g," ")}</td>
                              <td style={{padding:"6px 10px",color:THEME.textSecondary}}>{e.mae}</td>
                              <td style={{padding:"6px 10px",color:THEME.textSecondary}}>{e.rmse}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      </div>
                    </Card>
                  )}
                </>
              )}
            </>
          )}

          {/* ── WEIGHTS ── */}
          {tab==="weights"&&results&&(
            <Card title={`Allocation dynamique des poids - ${results.label}`} style={{background:"#a8a8a8"}} onExport={()=>csvDownload(weightData,`poids_${results.label}.csv`)}>
              <div style={{background:"#fff",borderRadius:8,padding:"8px 8px 4px 8px"}}>
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={weightData} stackOffset="expand" margin={{top:4,right:10,left:0,bottom:0}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={THEME.grid}/>
                  <XAxis dataKey="time" stroke={THEME.textMuted} height={28}
                    interval={Math.ceil(weightData.length/6)}
                    tickFormatter={v=>{const d=new Date(v);return d.toLocaleDateString("fr-FR",{month:"short"});}}
                    tick={{fontSize:10,fill:THEME.textSecondary}}/>
                  <YAxis stroke={THEME.textMuted} tick={{fontSize:9}} tickFormatter={v=>`${(v*100).toFixed(0)}%`}/>
                  <Tooltip labelFormatter={v=>new Date(v).toLocaleDateString("fr-FR",{day:"2-digit",month:"short",year:"numeric"})} formatter={v=>`${(v*100).toFixed(1)}%`} contentStyle={{background:"#e8e8e8",border:"1px solid #c8c8c8",borderRadius:8,fontSize:10,color:"#0e2d52"}}/>
                  <Legend wrapperStyle={{fontSize:10}}/>
                  {results.experts.map((e,i)=>(
                    <Area key={e} type="monotone" dataKey={e} stackId="1" stroke={PALETTE[i%PALETTE.length]} fill={PALETTE[i%PALETTE.length]} fillOpacity={0.82} name={e.replace(/rand_expert_/,"R-Exp ").replace(/_/g," ")}/>
                  ))}
                </AreaChart>
              </ResponsiveContainer>
              </div>

              {weightData.some(r=>r.wind_global_index!==undefined)&&(
                <div style={{marginTop:18}}>
                  <div style={{fontSize:11,fontWeight:700,color:"#000",marginBottom:8}}>Wind Global Index</div>
                  <div style={{background:"#fff",borderRadius:8,padding:"8px 8px 4px 8px"}}>
                    <ResponsiveContainer width="100%" height={140}>
                      <AreaChart data={weightData} margin={{top:4,right:10,left:0,bottom:0}}>
                        <defs><linearGradient id="wgiGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#2c8ca2" stopOpacity={0.62}/><stop offset="95%" stopColor="#2c8ca2" stopOpacity={0.06}/></linearGradient></defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={THEME.grid}/>
                        <XAxis dataKey="time" stroke={THEME.textMuted} height={28}
                          interval={Math.ceil(weightData.length/6)}
                          tickFormatter={v=>{const d=new Date(v);return d.toLocaleDateString("fr-FR",{month:"short"});}}
                          tick={{fontSize:10,fill:THEME.textSecondary}}/>
                        <YAxis stroke={THEME.textMuted} tick={{fontSize:9}} width={42}/>
                        <Tooltip labelFormatter={v=>new Date(v).toLocaleDateString("fr-FR",{day:"2-digit",month:"short",year:"numeric"})} formatter={v=>[v?.toFixed?.(3)??v,"Wind Global Index"]} contentStyle={{background:"#e8e8e8",border:"1px solid #c8c8c8",borderRadius:8,fontSize:10,color:"#0e2d52"}}/>
                        <Area type="monotone" dataKey="wind_global_index" stroke="#2c8ca2" fill="url(#wgiGrad)" strokeWidth={1.8} dot={false} name="Wind Global Index"/>
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              <div style={{display:"flex",flexWrap:"wrap",gap:8,marginTop:16}}>
                {results.experts.map((e,i)=>{
                  const avg=results.weightHistory.reduce((s,w)=>s+w[i],0)/results.weightHistory.length;
                  const mn=Math.min(...results.weightHistory.map(w=>w[i]));
                  const mx=Math.max(...results.weightHistory.map(w=>w[i]));
                  return(
                    <div key={e} style={{background:"#a8a8a8",borderRadius:8,padding:"6px",minWidth:130}}>
                      <div style={{background:"#fff",borderRadius:6,padding:"8px 10px"}}>
                        <div style={{fontSize:9,color:PALETTE[i%PALETTE.length],fontWeight:700,marginBottom:6}}>{e.replace(/rand_expert_/,"R-Exp ").replace(/_/g," ")}</div>
                        <div style={{fontSize:13,fontWeight:700,color:THEME.textPrimary}}>{(avg*100).toFixed(1)}%</div>
                        <div style={{fontSize:9,color:THEME.textMuted}}>min {(mn*100).toFixed(1)}% · max {(mx*100).toFixed(1)}%</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </Card>
          )}
          {tab==="weights"&&!results&&<div style={{textAlign:"center",color:"#dbe7ff",padding:80,fontSize:13}}>Lancez une exécution pour voir les poids dynamiques.</div>}

          {/* ── COMPARE ── */}
          {tab==="montecarlo"&&(
            <>
              <Card title="Simulation de Monte Carlo" style={{background:"#a8a8a8"}}>
                <div style={{display:"grid",gridTemplateColumns:"minmax(260px,0.9fr) minmax(320px,1.1fr)",gap:12}}>
                  <div style={{background:"#fff",borderRadius:8,padding:"10px 12px"}}>
                    <div style={{fontSize:10,color:THEME.textMuted,fontWeight:700,marginBottom:6,textTransform:"uppercase"}}>Paramètres de simulation</div>
                    <div style={{marginBottom:12}}>
                      <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
                        <span style={{fontSize:11,color:"#000",fontWeight:700}}>Nombre de simulations</span>
                        <span style={{fontSize:11,color:"#000"}}>{monteCarloCount}</span>
                      </div>
                      <input type="number" min={3} step={1} value={monteCarloCount} onChange={e=>setMonteCarloCount(Math.max(0,Math.round(Number(e.target.value)||0)))}
                        style={{width:"100%",border:`1px solid ${THEME.border}`,borderRadius:7,padding:"8px 10px",fontSize:12,boxSizing:"border-box",marginBottom:8}}/>
                      <div style={{fontSize:10,color:THEME.textSecondary}}>
                        Estimation initiale : <strong style={{color:"#0e2d52"}}>{lastRandomSetup?formatDuration(monteCarloEstimateMs):"en attente d'un tirage aléatoire"}</strong>
                      </div>
                      {lastRandomSetup&&(
                        <div style={{fontSize:10,color:THEME.textMuted,marginTop:4}}>
                          Base estimée sur {lastRandomSetup.rowCount} lignes, {lastRandomSetup.nExperts} experts, {monteCarloAlgoIds.length} méthodes et les paramètres figés du dernier clic sur « Générer ».
                        </div>
                      )}
                    </div>

                    <div style={{background:"#f4f7fb",border:`1px solid ${THEME.border}`,borderRadius:8,padding:"10px 10px 8px 10px",marginBottom:12}}>
                      <div style={{fontSize:10,color:"#0e2d52",fontWeight:700,marginBottom:6,textTransform:"uppercase"}}>Conditions de génération réutilisées</div>
                      {!lastRandomSetup&&<div style={{fontSize:10,color:THEME.textMuted}}>La simulation reprendra exactement les paramètres du dernier clic sur « Générer X experts » en mode aléatoire.</div>}
                      {lastRandomSetup&&(
                        <>
                          <div style={{display:"flex",flexWrap:"wrap",gap:6,marginBottom:6}}>
                            {[
                              `${lastRandomSetup.nExperts} experts`,
                              `phases ${lastRandomSetup.phaseMin}-${lastRandomSetup.phaseMax}`,
                              `bruit ${(lastRandomSetup.noiseLevel*100).toFixed(0)}%`,
                              `${lastRandomSetup.rowCount} lignes`,
                            ].map(tag=>(
                              <span key={tag} style={{background:"#dde7f4",color:"#0e2d52",borderRadius:999,padding:"4px 8px",fontSize:10,fontWeight:700}}>{tag}</span>
                            ))}
                          </div>
                          <div style={{fontSize:10,color:THEME.textSecondary}}>
                            Fenêtre figée : {lastRandomSetup.dateFrom||"min"} → {lastRandomSetup.dateTo||"max"}
                          </div>
                          {randomConfigDirty&&<div style={{fontSize:10,color:"#c0392b",marginTop:6}}>Les curseurs ou la fenêtre affichés à gauche ont changé depuis ce dernier tirage. Recliquez sur « Générer X experts » pour mettre la simulation à jour.</div>}
                        </>
                      )}
                    </div>

                    <div style={{display:"flex",gap:8,marginBottom:10}}>
                      <button onClick={handleMonteCarloRun} disabled={!monteCarloCanRun}
                        style={{flex:1,background:monteCarloCanRun?"#E2001A":THEME.border,color:"#fff",border:"none",borderRadius:8,padding:"10px 12px",fontSize:12,fontWeight:700,cursor:monteCarloCanRun?"pointer":"not-allowed"}}>
                        {monteCarloState.running?"Simulation en cours...":"Lancer"}
                      </button>
                      {monteCarloState.running&&(
                        <button onClick={handleStopMonteCarlo}
                          style={{background:"#E2001A",color:"#fff",border:"none",borderRadius:8,padding:"10px 12px",fontSize:11,fontWeight:700,cursor:"pointer"}}>
                          {monteCarloState.cancelRequested?"Arrêt...":"Stop"}
                        </button>
                      )}
                      <button onClick={()=>setMonteCarloAlgoIds(ALGOS.map(algo=>algo.id))} disabled={monteCarloState.running}
                        style={{background:"#fff",color:monteCarloState.running?THEME.textMuted:"#0e2d52",border:`1px solid ${THEME.border}`,borderRadius:8,padding:"10px 12px",fontSize:11,fontWeight:700,cursor:monteCarloState.running?"not-allowed":"pointer"}}>
                        Tout sélectionner
                      </button>
                      <button onClick={()=>setMonteCarloAlgoIds(["BOA","SimpleMean"])} disabled={monteCarloState.running}
                        style={{background:"#fff",color:monteCarloState.running?THEME.textMuted:"#0e2d52",border:`1px solid ${THEME.border}`,borderRadius:8,padding:"10px 12px",fontSize:11,fontWeight:700,cursor:monteCarloState.running?"not-allowed":"pointer"}}>
                        Tout désélectionner
                      </button>
                    </div>

                    {monteCarloWarnings.map(msg=>(
                      <div key={msg} style={{fontSize:10,color:"#c0392b",marginBottom:4}}>{msg}</div>
                    ))}
                    {!monteCarloWarnings.length&&<div style={{fontSize:10,color:"#1d7a5a"}}>La simulation se fera {monteCarloCount} fois dans les mêmes conditions que le dernier tirage aléatoire enregistré.</div>}
                    {!monteCarloWarnings.length&&(
                      <div style={{fontSize:10,color:THEME.textSecondary,marginTop:6,lineHeight:1.45,fontStyle:"italic"}}>
                        Chaque methode reprend son dernier parametrage propre; si elle n'a jamais ete lancee, ses valeurs par defaut sont utilisees.
                        {monteCarloSharedHmoeAlgoLabel?` Les regimes HMOE sont partages depuis le dernier run HMOE (${monteCarloSharedHmoeAlgoLabel}).`:" Les regimes HMOE restent sur la selection par defaut tant qu'aucun HMOE n'a ete lance."}
                      </div>
                    )}
                    {monteCarloState.error&&<div style={{fontSize:10,color:"#c0392b",marginTop:6}}>{monteCarloState.error}</div>}
                  </div>

                  <div style={{background:"#fff",borderRadius:8,padding:"10px 12px"}}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
                      <div style={{fontSize:10,color:THEME.textMuted,fontWeight:700,textTransform:"uppercase"}}>Méthodes d'agrégation ({monteCarloAlgoIds.length}/{ALGOS.length})</div>
                      <button onClick={()=>setMonteCarloAlgoIds(["BOA","SimpleMean"])} disabled={monteCarloState.running}
                        style={{background:"transparent",color:THEME.textMuted,border:`1px dashed ${THEME.border}`,borderRadius:8,padding:"4px 8px",fontSize:10,cursor:monteCarloState.running?"not-allowed":"pointer"}}>
                        Réinitialiser
                      </button>
                    </div>
                    <div style={{background:"#fff5f5",border:"1px solid #f1c4c4",borderRadius:8,padding:"8px 10px",marginBottom:10}}>
                      <div style={{fontSize:9,color:"#7a1f1f",fontWeight:800,textTransform:"uppercase",marginBottom:4}}>Condition de parametres</div>
                      <div style={{fontSize:10,color:"#7a1f1f",lineHeight:1.4}}>
                        Pour qu'un parametrage soit pris en compte ici, il faut d'abord lancer l'algo avec ces parametres.
                        Exception: la selection des regimes HMOE se propage a tous les HMOE depuis le dernier run HMOE.
                      </div>
                    </div>
                    <div style={{display:"flex",flexDirection:"column",gap:10}}>
                      {ALGO_GROUPS.map(group=>(
                        <div key={group.label} style={{background:"#f7f9fc",border:`1px solid ${THEME.border}`,borderRadius:8,padding:"8px 10px"}}>
                          <div style={{fontSize:10,color:"#0e2d52",fontWeight:800,marginBottom:6,textTransform:"uppercase"}}>{group.label}</div>
                          <div style={{display:"flex",flexWrap:"wrap",gap:8}}>
                            {group.algos.map(algo=>{
                              const selected=monteCarloAlgoIds.includes(algo.id);
                              const algoConfig=monteCarloAlgoRunConfigs[algo.id];
                              const paramTokens=getMonteCarloAlgoParamTokens(algo.id,algoConfig);
                              const hmoeSourceAlgoName=algoConfig.sharedHmoeSourceAlgoId?(ALGOS.find(candidate=>candidate.id===algoConfig.sharedHmoeSourceAlgoId)?.name||algoConfig.sharedHmoeSourceAlgoId):null;
                              return(
                                <div key={algo.id} style={{
                                  flex:"1 1 220px",minWidth:220,background:selected?"#fff4f4":"#ffffff",
                                  border:`1px solid ${selected?"#E2001A":"#d5e0ec"}`,borderRadius:10,padding:8,opacity:monteCarloState.running?0.78:1
                                }}>
                                  <button onClick={()=>toggleMonteCarloAlgo(algo.id)} disabled={monteCarloState.running} style={{
                                    width:"100%",background:selected?"#E2001A22":"#ffffff",color:selected?"#b42318":"#0e2d52",
                                    border:`1px solid ${selected?"#E2001A":"#c7d6e6"}`,borderRadius:8,padding:"6px 10px",fontSize:11,cursor:monteCarloState.running?"not-allowed":"pointer",fontWeight:selected?700:500,textAlign:"left"
                                  }}>
                                    {algo.name}
                                  </button>
                                  <div style={{marginTop:7,display:"flex",flexDirection:"column",gap:5}}>
                                    <div style={{fontSize:9,color:selected?"#b42318":THEME.textMuted,fontWeight:800,textTransform:"uppercase"}}>
                                      {getParamSourceLabel(algoConfig.source)}
                                    </div>
                                    {paramTokens.length>0?(
                                      <div style={{display:"flex",flexWrap:"wrap",gap:4}}>
                                        {paramTokens.map(token=>(
                                          <span key={token} style={{background:selected?"#fde2e2":"#eef3f8",color:"#0e2d52",borderRadius:999,padding:"3px 7px",fontSize:9,fontWeight:700}}>
                                            {token}
                                          </span>
                                        ))}
                                      </div>
                                    ):(
                                      <div style={{fontSize:10,color:THEME.textSecondary}}>Aucun parametre specifique.</div>
                                    )}
                                    {HMOE_ALGO_IDS.includes(algo.id)&&(
                                      <>
                                        <div style={{fontSize:10,color:THEME.textSecondary,lineHeight:1.45}}>
                                          Regimes HMOE
                                        </div>
                                        <div style={{display:"flex",flexWrap:"wrap",gap:4}}>
                                          {getHmoeRegimeNames(algoConfig.selectedHmoeRegimes).map(regimeName=>(
                                            <span key={regimeName} style={{background:"#fff1f1",color:"#9f1d1d",border:"1px solid #efb7b7",borderRadius:999,padding:"3px 8px",fontSize:9,fontWeight:700}}>
                                              {regimeName}
                                            </span>
                                          ))}
                                        </div>
                                        <div style={{fontSize:9,color:algoConfig.regimesSource==="shared-hmoe-run"?"#7a1f1f":THEME.textMuted,lineHeight:1.35,fontStyle:algoConfig.regimesSource==="shared-hmoe-run"?"italic":"normal",fontWeight:algoConfig.regimesSource==="shared-hmoe-run"?600:400}}>
                                          {getHmoeRegimeSourceLabel(algoConfig.regimesSource,hmoeSourceAlgoName)}
                                        </div>
                                      </>
                                    )}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </Card>

              {monteCarloState.running&&(
                <Card title="Progression de la simulation" style={{background:"#a8a8a8"}}>
                  <div style={{background:"#fff",borderRadius:8,padding:"12px"}}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
                      <div style={{fontSize:12,fontWeight:700,color:"#0e2d52"}}>
                        {Math.round((monteCarloState.progress||0)*100)}% terminé
                      </div>
                      <div style={{fontSize:10,color:THEME.textMuted}}>
                        {monteCarloCurrentAlgoLabel?`Simulation ${Math.min(monteCarloState.simulationIndex+1,monteCarloCount)}/${monteCarloCount} · ${monteCarloCurrentAlgoLabel}`:`Simulation ${Math.min(monteCarloState.simulationIndex+1,monteCarloCount)}/${monteCarloCount}`}
                      </div>
                    </div>
                    <div style={{height:12,background:"#dde7f4",borderRadius:999,overflow:"hidden",marginBottom:8}}>
                      <div style={{width:`${Math.max(2,(monteCarloState.progress||0)*100)}%`,height:"100%",background:"linear-gradient(90deg,#E2001A 0%,#ff7a7a 100%)",borderRadius:999,transition:"width .2s ease"}}/>
                    </div>
                    <div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:THEME.textSecondary}}>
                      <span>Temps écoulé : {formatDuration(monteCarloState.elapsedMs)}</span>
                      <span>Temps restant estimé : {formatDuration(monteCarloState.remainingMs)}</span>
                    </div>
                  </div>
                </Card>
              )}

              {!monteCarloResult&&!monteCarloState.running&&(
                <div style={{textAlign:"center",color:"#dbe7ff",padding:80,fontSize:13}}>Configurez votre simulation puis lancez-la pour obtenir les classements n-avg.</div>
              )}

              {monteCarloResult&&monteCarloResult.rankings&&(
                <>
                  <div style={{display:"flex",gap:10}}>
                    {[{label:"Simulations",val:monteCarloResult.simulationCount,c:"#7f64c3"},{label:"Méthodes comparées",val:monteCarloResult.averages.length,c:"#4c72b8"},{label:"Lignes par run",val:monteCarloResult.rowCount,c:"#2f8d73"},{label:"Temps estimé",val:formatDuration(monteCarloResult.estimatedMs),c:"#c69427"}].map(item=>(
                      <div key={item.label} style={{flex:1,background:"#a8a8a8",borderRadius:10,padding:"10px 14px"}}>
                        <div style={{fontSize:10,color:"#000",fontWeight:600,marginBottom:6}}>{item.label}</div>
                        <div style={{background:"#fff",borderRadius:6,padding:"6px 10px",display:"inline-block",minWidth:"100%",boxSizing:"border-box"}}>
                          <span style={{fontSize:18,fontWeight:700,color:item.c}}>{item.val}</span>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                    <Card title={`Classement n-avg général (n=${monteCarloResult.simulationCount})`} style={{gridColumn:"1/-1"}} onExport={()=>csvDownload(monteCarloResult.rankings.general.map((r,i)=>({rank:i+1,label:r.label,mae_avg:r.mae.toFixed(0),rmse_avg:r.rmse.toFixed(0),mape_avg:r.mape.toFixed(2)})),`mc_classement_general_n${monteCarloResult.simulationCount}.csv`)}>
                      <div style={{background:"#fff",borderRadius:8,padding:"8px"}}>
                        <div style={{display:"flex",gap:0,flexWrap:"wrap"}}>
                          {monteCarloResult.rankings.general.map((run,i)=>{
                            const col=MOE_PALETTE[Math.max(0,monteCarloResult.selectedAlgoIds.indexOf(run.id))%MOE_PALETTE.length];
                            return(
                              <div key={run.id} style={{flex:"1 1 160px",background:"transparent",border:`${i===0?"5px solid #fbbf24":`2px solid ${col}66`}`,borderRadius:10,padding:"12px 14px",margin:4}}>
                                <div style={{fontSize:20,marginBottom:2,color:"#000"}}>#{i+1}</div>
                                <div style={{fontSize:12,fontWeight:700,color:col,marginBottom:4}}>{run.label}</div>
                                <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
                                  {[{k:"MAE",v:run.mae.toFixed(0),u:"MW"},{k:"RMSE",v:run.rmse.toFixed(0),u:"MW"},{k:"MAPE",v:run.mape.toFixed(2),u:"%"}].map(metric=>(
                                    <div key={metric.k} style={{padding:"3px 7px",fontSize:10}}><span style={{color:"#000"}}>{metric.k} </span><span style={{color:"#000",fontWeight:600}}>{metric.v}{metric.u}</span></div>
                                  ))}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </Card>
                    {[{title:"Classement n-avg MAE",key:"byMAE",metric:"mae",unit:"MW",color:"#4c72b8"},{title:"Classement n-avg RMSE",key:"byRMSE",metric:"rmse",unit:"MW",color:"#2f8d73"},{title:"Classement n-avg MAPE",key:"byMAPE",metric:"mape",unit:"%",color:"#c69427"}].map(({title,key,metric,unit,color})=>(
                      <Card key={key} title={title} onExport={()=>csvDownload(monteCarloResult.rankings[key].map((r,i)=>({rank:i+1,label:r.label,[metric+"_avg"]:r[metric].toFixed(metric==="mape"?2:0)})),`mc_${key}_n${monteCarloResult.simulationCount}.csv`)}>
                        <div style={{background:"#fff",borderRadius:8,padding:"8px"}}>
                          <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                            <thead><tr style={{borderBottom:`1px solid ${THEME.border}`}}>{["Rang","Méthode",metric.toUpperCase()].map(h=><th key={h} style={{textAlign:"left",padding:"5px 8px",color:THEME.textMuted,fontWeight:600}}>{h}</th>)}</tr></thead>
                            <tbody>
                              {monteCarloResult.rankings[key].map((run,i)=>{
                                const col=MOE_PALETTE[Math.max(0,monteCarloResult.selectedAlgoIds.indexOf(run.id))%MOE_PALETTE.length];
                                const val=metric==="mape"?run.mape.toFixed(2):run[metric].toFixed(0);
                                const pct=Math.max(10,(run[metric]/Math.max(...monteCarloResult.rankings[key].map(r=>r[metric])))*100);
                                return(
                                  <tr key={run.id} style={{borderBottom:`1px solid ${THEME.border}`}}>
                                    <td style={{padding:"6px 8px",fontWeight:700,color:medalColor(i+1),fontSize:13}}>#{i+1}</td>
                                    <td style={{padding:"6px 8px"}}>
                                      <div style={{display:"flex",alignItems:"center",gap:6}}><span style={{width:8,height:8,borderRadius:"50%",background:col,display:"inline-block"}}/><span style={{color:col,fontWeight:600}}>{run.label}</span></div>
                                      <div style={{marginTop:3,height:3,borderRadius:2,background:THEME.panelBg,overflow:"hidden"}}><div style={{width:`${pct}%`,height:"100%",background:color,borderRadius:2}}/></div>
                                    </td>
                                    <td style={{padding:"6px 8px",textAlign:"right",fontWeight:i===0?700:400,color:i===0?color:THEME.textSecondary}}>{val}{unit}</td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </Card>
                    ))}
                  </div>
                </>
              )}
            </>
          )}

          {tab==="gridsearchmc"&&(
            <>
              <Card title="Monte Carlo Gridsearch par méthode" style={{background:"#a8a8a8"}}>
                <div style={{display:"grid",gridTemplateColumns:"minmax(260px,0.92fr) minmax(360px,1.08fr)",gap:12}}>
                  <div style={{background:"#fff",borderRadius:8,padding:"10px 12px"}}>
                    <div style={{fontSize:10,color:THEME.textMuted,fontWeight:700,marginBottom:6,textTransform:"uppercase"}}>Paramètres de simulation</div>
                    <div style={{marginBottom:12}}>
                      <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
                        <span style={{fontSize:11,color:"#000",fontWeight:700}}>Nombre de simulations</span>
                        <span style={{fontSize:11,color:"#000"}}>{gridSearchSimulationCount}</span>
                      </div>
                      <input type="number" min={3} step={1} value={gridSearchSimulationCount} onChange={e=>setGridSearchSimulationCount(Math.max(0,Math.round(Number(e.target.value)||0)))}
                        style={{width:"100%",border:`1px solid ${THEME.border}`,borderRadius:7,padding:"8px 10px",fontSize:12,boxSizing:"border-box",marginBottom:8}}/>
                      <div style={{fontSize:10,color:THEME.textSecondary}}>
                        Estimation initiale : <strong style={{color:"#0e2d52"}}>{lastRandomSetup?formatDuration(gridSearchEstimateMs):"en attente d'un tirage aléatoire"}</strong>
                      </div>
                      {lastRandomSetup&&(
                        <div style={{fontSize:10,color:THEME.textMuted,marginTop:4}}>
                          Base estimée sur {lastRandomSetup.rowCount} lignes, {lastRandomSetup.nExperts} experts, {gridSearchCombos.length} combinaisons et la complexité de {gridSearchSelectedAlgo?.name||gridSearchAlgoId}.
                        </div>
                      )}
                    </div>

                    <div style={{background:"#f4f7fb",border:`1px solid ${THEME.border}`,borderRadius:8,padding:"10px 10px 8px 10px",marginBottom:12}}>
                      <div style={{fontSize:10,color:"#0e2d52",fontWeight:700,marginBottom:6,textTransform:"uppercase"}}>Conditions de génération réutilisées</div>
                      {!lastRandomSetup&&<div style={{fontSize:10,color:THEME.textMuted}}>Le gridsearch reprendra exactement les paramètres du dernier clic sur « Générer X experts » en mode aléatoire.</div>}
                      {lastRandomSetup&&(
                        <>
                          <div style={{display:"flex",flexWrap:"wrap",gap:6,marginBottom:6}}>
                            {[
                              `${lastRandomSetup.nExperts} experts`,
                              `phases ${lastRandomSetup.phaseMin}-${lastRandomSetup.phaseMax}`,
                              `bruit ${(lastRandomSetup.noiseLevel*100).toFixed(0)}%`,
                              `${lastRandomSetup.rowCount} lignes`,
                            ].map(tag=>(
                              <span key={tag} style={{background:"#dde7f4",color:"#0e2d52",borderRadius:999,padding:"4px 8px",fontSize:10,fontWeight:700}}>{tag}</span>
                            ))}
                          </div>
                          <div style={{fontSize:10,color:THEME.textSecondary}}>
                            Fenêtre figée : {lastRandomSetup.dateFrom||"min"} → {lastRandomSetup.dateTo||"max"}
                          </div>
                          {randomConfigDirty&&<div style={{fontSize:10,color:"#c0392b",marginTop:6}}>Les curseurs ou la fenêtre affichés à gauche ont changé depuis ce dernier tirage. Recliquez sur « Générer X experts » pour mettre la simulation à jour.</div>}
                        </>
                      )}
                    </div>

                    <div style={{display:"flex",gap:8,marginBottom:10}}>
                      <button onClick={handleGridSearchRun} disabled={!gridSearchCanRun}
                        style={{flex:1,background:gridSearchCanRun?"#E2001A":THEME.border,color:"#fff",border:"none",borderRadius:8,padding:"10px 12px",fontSize:12,fontWeight:700,cursor:gridSearchCanRun?"pointer":"not-allowed"}}>
                        {gridSearchState.running?"Gridsearch en cours...":"Lancer"}
                      </button>
                      {gridSearchState.running&&(
                        <button onClick={handleStopGridSearch}
                          style={{background:"#E2001A",color:"#fff",border:"none",borderRadius:8,padding:"10px 12px",fontSize:11,fontWeight:700,cursor:"pointer"}}>
                          {gridSearchState.cancelRequested?"Arrêt...":"Stop"}
                        </button>
                      )}
                      <button onClick={()=>{
                        setGridSearchCombos(buildGridSearchCombosForAlgo(gridSearchAlgoId));
                        setGridSearchResult(null);
                        setGridSearchState(createGridSearchAsyncState());
                      }} disabled={gridSearchState.running}
                        style={{background:"#fff",color:gridSearchState.running?THEME.textMuted:"#0e2d52",border:`1px solid ${THEME.border}`,borderRadius:8,padding:"10px 12px",fontSize:11,fontWeight:700,cursor:gridSearchState.running?"not-allowed":"pointer"}}>
                        Réinitialiser
                      </button>
                    </div>
                    <div style={{fontSize:10,color:THEME.textSecondary,marginBottom:8,lineHeight:1.4,fontStyle:"italic"}}>Cette page fait s'affronter une seule méthode contre elle-même avec plusieurs jeux de paramètres, puis calcule les moyennes n-avg sur MAE, RMSE et MAPE.</div>

                    {gridSearchWarnings.map(msg=>(
                      <div key={msg} style={{fontSize:10,color:"#c0392b",marginBottom:4}}>{msg}</div>
                    ))}
                    {!gridSearchWarnings.length&&<div style={{fontSize:10,color:"#1d7a5a"}}>{gridSearchCombos.length} combinaisons distinctes seront relancées {gridSearchSimulationCount} fois dans les mêmes conditions aléatoires figées.</div>}
                    {gridSearchState.error&&<div style={{fontSize:10,color:"#c0392b",marginTop:6}}>{gridSearchState.error}</div>}
                  </div>

                  <div style={{background:"#fff",borderRadius:8,padding:"10px 12px"}}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
                      <div style={{fontSize:10,color:THEME.textMuted,fontWeight:700,textTransform:"uppercase"}}>Méthode et combinaisons ({gridSearchCombos.length})</div>
                      <button onClick={addGridSearchCombo} disabled={gridSearchState.running}
                        style={{background:"transparent",color:gridSearchState.running?THEME.textMuted:"#0e2d52",border:`1px dashed ${THEME.border}`,borderRadius:8,padding:"4px 8px",fontSize:10,cursor:gridSearchState.running?"not-allowed":"pointer"}}>
                        + Ajouter une combinaison
                      </button>
                    </div>

                    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:12}}>
                      <div>
                        <div style={{fontSize:10,color:THEME.textMuted,fontWeight:700,marginBottom:4}}>Famille</div>
                        <select value={gridSearchGroupLabel} onChange={e=>handleGridSearchGroupChange(e.target.value)}
                          style={{width:"100%",border:`1px solid ${THEME.border}`,borderRadius:7,padding:"8px 10px",fontSize:11,boxSizing:"border-box",background:"#fff"}}>
                          {ALGO_GROUPS.map(group=><option key={group.label} value={group.label}>{group.label}</option>)}
                        </select>
                      </div>
                      <div>
                        <div style={{fontSize:10,color:THEME.textMuted,fontWeight:700,marginBottom:4}}>Méthode</div>
                        <select value={gridSearchAlgoId} onChange={e=>handleGridSearchAlgoChange(e.target.value)}
                          style={{width:"100%",border:`1px solid ${THEME.border}`,borderRadius:7,padding:"8px 10px",fontSize:11,boxSizing:"border-box",background:"#fff"}}>
                          {gridSearchAvailableAlgos.map(algo=><option key={algo.id} value={algo.id}>{algo.name}</option>)}
                        </select>
                      </div>
                    </div>

                    <div style={{fontSize:10,color:THEME.textSecondary,marginBottom:10}}>
                      Classement interne de <strong style={{color:"#0e2d52"}}>{gridSearchSelectedAlgo?.name||gridSearchAlgoId}</strong> selon plusieurs combinaisons de paramètres.
                    </div>

                    <div style={{display:"flex",flexDirection:"column",gap:10,maxHeight:620,overflowY:"auto",paddingRight:2}}>
                      {gridSearchCombos.map((combo,index)=>{
                        const comboLabel=buildGridSearchComboLabel(gridSearchAlgoId,combo,index);
                        return(
                          <div key={combo.id} style={{background:"#f7f9fc",border:`1px solid ${THEME.border}`,borderRadius:10,padding:"10px 10px 8px 10px"}}>
                            <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",gap:10,marginBottom:8}}>
                              <div>
                                <div style={{fontSize:11,fontWeight:800,color:"#0e2d52"}}>{getGridSearchComboDisplayTitle(index)}</div>
                                <div style={{fontSize:10,color:THEME.textMuted,marginTop:2}}>{comboLabel}</div>
                              </div>
                              <button onClick={()=>removeGridSearchCombo(combo.id)} disabled={gridSearchState.running||gridSearchCombos.length<=1}
                                style={{background:"transparent",color:gridSearchState.running||gridSearchCombos.length<=1?THEME.textMuted:"#b42318",border:`1px solid ${gridSearchState.running||gridSearchCombos.length<=1?THEME.border:"#e2a7a7"}`,borderRadius:8,padding:"4px 8px",fontSize:10,cursor:gridSearchState.running||gridSearchCombos.length<=1?"not-allowed":"pointer"}}>
                                Supprimer
                              </button>
                            </div>

                            {gridSearchControlSections.length===0&&(
                              <div style={{fontSize:10,color:THEME.textSecondary}}>Cette méthode n'expose pas de paramètre variable dans l'application actuelle.</div>
                            )}

                            {gridSearchControlSections.map(section=>(
                              <div key={`${combo.id}-${section.id}`} style={{background:"#fff",border:`1px solid ${THEME.border}`,borderRadius:8,padding:"8px 10px",marginBottom:8}}>
                                <div style={{fontSize:10,color:"#0e2d52",fontWeight:800,textTransform:"uppercase",marginBottom:6}}>{section.title}</div>

                                {section.controls.some(control=>control.type!=="multiselect")&&(
                                  <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(170px,1fr))",gap:8}}>
                                    {section.controls.filter(control=>control.type!=="multiselect").map(control=>{
                                      const controlValue=control.scope==="root"?combo[control.id]:combo[control.scope][control.id];
                                      return(
                                        <div key={`${combo.id}-${control.scope}-${control.id}`}>
                                          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:4}}>
                                            <span style={{fontSize:10,color:THEME.textSecondary,fontWeight:700}}>{control.label}</span>
                                            {control.type==="slider"&&<span style={{fontSize:10,color:"#0e2d52",fontWeight:700}}>{controlValue}</span>}
                                          </div>

                                          {control.type==="select"&&(
                                            <select value={controlValue} onChange={e=>{
                                              const nextValue=typeof control.options?.[0]?.value==="number"?parseFloat(e.target.value):e.target.value;
                                              updateGridSearchControl(combo.id,control.scope,control.id,nextValue);
                                            }}
                                              style={{width:"100%",border:`1px solid ${THEME.border}`,borderRadius:7,padding:"7px 8px",fontSize:11,boxSizing:"border-box",background:"#fff"}}>
                                              {control.options.map(option=><option key={option.value} value={option.value}>{option.label}</option>)}
                                            </select>
                                          )}

                                          {control.type==="slider"&&(
                                            <input type="range" min={control.min} max={control.max} step={control.step} value={controlValue}
                                              onChange={e=>updateGridSearchControl(combo.id,control.scope,control.id,+e.target.value)}
                                              style={{width:"100%",accentColor:"#E2001A"}}/>
                                          )}

                                          {control.type==="toggle"&&(
                                            <button onClick={()=>updateGridSearchControl(combo.id,control.scope,control.id,!controlValue)}
                                              style={{width:"100%",display:"flex",justifyContent:"space-between",alignItems:"center",background:controlValue?"#eef6ff":"#f8fafc",border:`1px solid ${controlValue?"#b6d1f3":THEME.border}`,borderRadius:8,padding:"8px 10px",fontSize:11,color:"#0e2d52",cursor:"pointer"}}>
                                              <span>{controlValue?"Activé":"Désactivé"}</span>
                                              <span style={{fontWeight:800,color:controlValue?"#1d4ed8":"#64748b"}}>{controlValue?"ON":"OFF"}</span>
                                            </button>
                                          )}
                                        </div>
                                      );
                                    })}
                                  </div>
                                )}

                                {section.controls.filter(control=>control.type==="multiselect").map(control=>(
                                  <div key={`${combo.id}-${control.id}`} style={{display:"flex",flexDirection:"column",gap:6}}>
                                    <div style={{display:"flex",flexWrap:"wrap",gap:6}}>
                                      {control.options.map(option=>{
                                        const selected=combo.selectedHmoeRegimes.includes(option.value);
                                        return(
                                          <button key={option.value} onClick={()=>toggleGridSearchRegime(combo.id,option.value)} style={{
                                            background:selected?"#fff1f1":"#ffffff",
                                            color:selected?"#9f1d1d":"#0e2d52",
                                            border:`1px solid ${selected?"#efb7b7":"#c7d6e6"}`,
                                            borderRadius:999,padding:"5px 9px",fontSize:10,cursor:"pointer",fontWeight:selected?700:500
                                          }}>
                                            {option.label}
                                          </button>
                                        );
                                      })}
                                    </div>
                                    <div style={{fontSize:9,color:THEME.textMuted}}>
                                      {combo.selectedHmoeRegimes.length} régime(s) actif(s)
                                    </div>
                                  </div>
                                ))}
                              </div>
                            ))}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </Card>

              {gridSearchState.running&&(
                <Card title="Progression du gridsearch" style={{background:"#a8a8a8"}}>
                  <div style={{background:"#fff",borderRadius:8,padding:"12px"}}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
                      <div style={{fontSize:12,fontWeight:700,color:"#0e2d52"}}>
                        {Math.round((gridSearchState.progress||0)*100)}% terminé
                      </div>
                      <div style={{fontSize:10,color:THEME.textMuted}}>
                        {gridSearchCurrentLabel?`Simulation ${Math.min(gridSearchState.simulationIndex+1,gridSearchSimulationCount)}/${gridSearchSimulationCount} · ${gridSearchCurrentLabel}`:`Simulation ${Math.min(gridSearchState.simulationIndex+1,gridSearchSimulationCount)}/${gridSearchSimulationCount}`}
                      </div>
                    </div>
                    <div style={{height:12,background:"#dde7f4",borderRadius:999,overflow:"hidden",marginBottom:8}}>
                      <div style={{width:`${Math.max(2,(gridSearchState.progress||0)*100)}%`,height:"100%",background:"linear-gradient(90deg,#E2001A 0%,#ff7a7a 100%)",borderRadius:999,transition:"width .2s ease"}}/>
                    </div>
                    <div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:THEME.textSecondary}}>
                      <span>Temps écoulé : {formatDuration(gridSearchState.elapsedMs)}</span>
                      <span>Temps restant estimé : {formatDuration(gridSearchState.remainingMs)}</span>
                    </div>
                  </div>
                </Card>
              )}

              {!gridSearchResult&&!gridSearchState.running&&(
                <div style={{textAlign:"center",color:"#dbe7ff",padding:80,fontSize:13}}>Configurez votre méthode, définissez au moins 2 combinaisons distinctes, puis lancez le gridsearch Monte Carlo.</div>
              )}

              {gridSearchResult&&gridSearchResult.rankings&&(
                <>
                  <div style={{display:"flex",gap:10}}>
                    {[{label:"Méthode",val:gridSearchResult.selectedAlgoLabel,c:"#b42318"},{label:"Simulations",val:gridSearchResult.simulationCount,c:"#7f64c3"},{label:"Combinaisons",val:gridSearchResult.averages.length,c:"#4c72b8"},{label:"Lignes par run",val:gridSearchResult.rowCount,c:"#2f8d73"},{label:"Temps estimé",val:formatDuration(gridSearchResult.estimatedMs),c:"#c69427"}].map(item=>(
                      <div key={item.label} style={{flex:1,background:"#a8a8a8",borderRadius:10,padding:"10px 14px"}}>
                        <div style={{fontSize:10,color:"#000",fontWeight:600,marginBottom:6}}>{item.label}</div>
                        <div style={{background:"#fff",borderRadius:6,padding:"6px 10px",display:"inline-block",minWidth:"100%",boxSizing:"border-box"}}>
                          <span style={{fontSize:16,fontWeight:700,color:item.c}}>{item.val}</span>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                    <Card title={`Classement n-avg général (n=${gridSearchResult.simulationCount})`} style={{gridColumn:"1/-1"}} onExport={()=>csvDownload(gridSearchResult.rankings.general.map((r,i)=>({rank:i+1,label:r.label,mae_avg:r.mae.toFixed(0),rmse_avg:r.rmse.toFixed(0),mape_avg:r.mape.toFixed(2)})),`gs_classement_general_n${gridSearchResult.simulationCount}.csv`)}>
                      <div style={{background:"#fff",borderRadius:8,padding:"8px"}}>
                        <div style={{display:"flex",gap:0,flexWrap:"wrap"}}>
                          {gridSearchResult.rankings.general.map((run,i)=>{
                            const col=MOE_PALETTE[Math.max(0,gridSearchResult.combos.findIndex(combo=>combo.id===run.id))%MOE_PALETTE.length];
                            return(
                              <div key={run.id} style={{flex:"1 1 190px",background:"transparent",border:`${i===0?"5px solid #fbbf24":`2px solid ${col}66`}`,borderRadius:10,padding:"12px 14px",margin:4}}>
                                <div style={{fontSize:20,marginBottom:2,color:"#000"}}>#{i+1}</div>
                                <div style={{fontSize:12,fontWeight:700,color:col,marginBottom:4}}>{run.label}</div>
                                <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
                                  {[{k:"MAE",v:run.mae.toFixed(0),u:"MW"},{k:"RMSE",v:run.rmse.toFixed(0),u:"MW"},{k:"MAPE",v:run.mape.toFixed(2),u:"%"}].map(metric=>(
                                    <div key={metric.k} style={{padding:"3px 7px",fontSize:10}}><span style={{color:"#000"}}>{metric.k} </span><span style={{color:"#000",fontWeight:600}}>{metric.v}{metric.u}</span></div>
                                  ))}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </Card>
                    {[{title:"Classement n-avg MAE",key:"byMAE",metric:"mae",unit:"MW",color:"#4c72b8"},{title:"Classement n-avg RMSE",key:"byRMSE",metric:"rmse",unit:"MW",color:"#2f8d73"},{title:"Classement n-avg MAPE",key:"byMAPE",metric:"mape",unit:"%",color:"#c69427"}].map(({title,key,metric,unit,color})=>(
                      <Card key={key} title={title} onExport={()=>csvDownload(gridSearchResult.rankings[key].map((r,i)=>({rank:i+1,label:r.label,[metric+"_avg"]:r[metric].toFixed(metric==="mape"?2:0)})),`gs_${key}_n${gridSearchResult.simulationCount}.csv`)}>
                        <div style={{background:"#fff",borderRadius:8,padding:"8px"}}>
                          <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                            <thead><tr style={{borderBottom:`1px solid ${THEME.border}`}}>{["Rang","Combinaison",metric.toUpperCase()].map(h=><th key={h} style={{textAlign:"left",padding:"5px 8px",color:THEME.textMuted,fontWeight:600}}>{h}</th>)}</tr></thead>
                            <tbody>
                              {gridSearchResult.rankings[key].map((run,i)=>{
                                const col=MOE_PALETTE[Math.max(0,gridSearchResult.combos.findIndex(combo=>combo.id===run.id))%MOE_PALETTE.length];
                                const val=metric==="mape"?run.mape.toFixed(2):run[metric].toFixed(0);
                                const pct=Math.max(10,(run[metric]/Math.max(...gridSearchResult.rankings[key].map(r=>r[metric])))*100);
                                return(
                                  <tr key={run.id} style={{borderBottom:`1px solid ${THEME.border}`}}>
                                    <td style={{padding:"6px 8px",fontWeight:700,color:medalColor(i+1),fontSize:13}}>#{i+1}</td>
                                    <td style={{padding:"6px 8px"}}>
                                      <div style={{display:"flex",alignItems:"center",gap:6}}><span style={{width:8,height:8,borderRadius:"50%",background:col,display:"inline-block"}}/><span style={{color:col,fontWeight:600}}>{run.label}</span></div>
                                      <div style={{marginTop:3,height:3,borderRadius:2,background:THEME.panelBg,overflow:"hidden"}}><div style={{width:`${pct}%`,height:"100%",background:color,borderRadius:2}}/></div>
                                    </td>
                                    <td style={{padding:"6px 8px",textAlign:"right",fontWeight:i===0?700:400,color:i===0?color:THEME.textSecondary}}>{val}{unit}</td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </Card>
                    ))}
                  </div>
                </>
              )}
            </>
          )}

          {tab==="compare"&&(
            <>
              {allRuns.length===0
                ?<div style={{textAlign:"center",color:"#dbe7ff",padding:80,fontSize:13}}>Aucune exécution enregistrée. Lancez au moins une exécution.</div>
                :(
                  <>
                    <Card title="Méthodes d'agrégation à afficher">
                      <div style={{background:"#fff",borderRadius:8,padding:"8px"}}>
                      <div style={{display:"flex",flexWrap:"wrap",gap:8}}>
                        {allRuns.map((run,i)=>{
                          const isVis=visibleRuns.has(run.id);const col=MOE_PALETTE[i%MOE_PALETTE.length];
                          return(
                            <div key={run.id} style={{position:"relative",display:"inline-flex"}}>
                              <button onClick={()=>{setVisibleRuns(prev=>{const next=new Set(prev);if(next.has(run.id))next.delete(run.id);else next.add(run.id);return next;});}} style={{
                                background:isVis?col+"22":"transparent",color:isVis?col:THEME.textDim,
                                border:`2px solid ${isVis?col:THEME.border}`,borderRadius:8,padding:"6px 14px 6px 10px",
                                fontSize:11,cursor:"pointer",fontWeight:isVis?700:400,display:"flex",alignItems:"center",gap:6
                              }}>
                                <span style={{width:8,height:8,borderRadius:"50%",background:isVis?col:THEME.textDim,display:"inline-block"}}/>
                                {run.label}
                                <span style={{fontSize:9,color:isVis?col+"aa":THEME.border,fontWeight:400}}>{run.expertMode}</span>
                              </button>
                              <span onClick={e=>{e.stopPropagation();setAllRuns(prev=>prev.filter(r=>r.id!==run.id));setVisibleRuns(prev=>{const next=new Set(prev);next.delete(run.id);return next;});}} style={{
                                position:"absolute",top:-6,left:-6,width:16,height:16,borderRadius:"50%",
                                background:"#c0392b",color:"#fff",fontSize:10,fontWeight:700,
                                display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer",lineHeight:1
                              }}>×</span>
                            </div>
                          );
                        })}
                        {allRuns.length>1&&<button onClick={()=>setVisibleRuns(new Set(allRuns.map(r=>r.id)))} style={{background:"transparent",color:THEME.textMuted,border:`1px dashed ${THEME.border}`,borderRadius:8,padding:"6px 12px",fontSize:11,cursor:"pointer"}}>Tout afficher</button>}
                        {allRuns.length>0&&<button onClick={()=>{setAllRuns([]);setVisibleRuns(new Set());}} style={{background:"transparent",color:"#c0392b",border:`1px dashed #c0392b`,borderRadius:8,padding:"6px 12px",fontSize:11,cursor:"pointer"}}>Vider</button>}
                      </div>
                      </div>
                    </Card>

                    {visibleRunsList.length>0&&(
                      <>
                        <Card title={`Comparaison des méthodes d'agrégation - ${cmpHorizon}h`} style={{background:"#a8a8a8"}} onExport={()=>csvDownload(cmpChartData,`comparaison_${cmpHorizon}h.csv`)}>
                          <div style={{display:"flex",justifyContent:"flex-end",gap:4,marginBottom:10}}>
                            {[24,48,72].map(h=>(
                              <button key={h} onClick={()=>setCmpHorizon(h)} style={{background:cmpHorizon===h?"#E2001A":"#fff",color:cmpHorizon===h?"#fff":"#0e2d52",border:`1px solid ${cmpHorizon===h?"#E2001A":"#b0c4d8"}`,borderRadius:7,padding:"4px 10px",fontSize:11,fontWeight:600,cursor:"pointer"}}>{h}h</button>
                            ))}
                          </div>
                          <div style={{background:"#fff",borderRadius:8,padding:"8px 8px 4px 8px"}}>
                          <ResponsiveContainer width="100%" height={320}>
                            <LineChart data={cmpChartData} margin={{top:4,right:10,left:0,bottom:0}}>
                              <CartesianGrid strokeDasharray="3 3" stroke={THEME.grid}/>
                              <XAxis dataKey="time" stroke={THEME.textMuted} height={36} interval={cmpHorizon===24?1:cmpHorizon===48?3:7} tickFormatter={v=>norm(v).slice(11,13)+"h"} tick={{fontSize:10,fill:THEME.textSecondary}}/>
                              <YAxis stroke={THEME.textMuted} tick={{fontSize:9}} unit=" MW" width={58}/>
                              <Tooltip contentStyle={{background:"#e8e8e8",border:"1px solid #c8c8c8",borderRadius:8,fontSize:10,color:"#0e2d52"}}/>
                              <Legend wrapperStyle={{fontSize:10}}/>
                              <Line type="monotone" dataKey="actual" dot={false} stroke="#173c66" strokeWidth={2.5} strokeDasharray="5 3" name="y_true"/>
                              {visibleRunsList.map((run,i)=>(
                                <Line key={run.id} type="monotone" dataKey={`moe_${run.label}`} dot={false} stroke={MOE_PALETTE[runIndex(run.id)%MOE_PALETTE.length]} strokeWidth={2} name={run.label}/>
                              ))}
                            </LineChart>
                          </ResponsiveContainer>
                          </div>
                        </Card>

                        {rankings&&(
                          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                            <Card title="Classement général" style={{gridColumn:"1/-1"}} onExport={()=>csvDownload(rankings.general.map((r,i)=>({rank:i+1,label:r.label,mae:r.mae.toFixed(0),rmse:r.rmse.toFixed(0),mape:r.mape.toFixed(2)})),"classement_general.csv")}>
                              <div style={{background:"#fff",borderRadius:8,padding:"8px"}}>
                              <div style={{display:"flex",gap:0,flexWrap:"wrap"}}>
                                {rankings.general.map((run,i)=>{
                                  const col=MOE_PALETTE[allRuns.indexOf(run)%MOE_PALETTE.length];
                                  return(
                                    <div key={run.id} style={{flex:"1 1 140px",background:"transparent",border:`${i===0?"5px solid #fbbf24":`2px solid ${col}66`}`,borderRadius:10,padding:"12px 14px",margin:4,position:"relative",overflow:"hidden"}}>
                                      <div style={{fontSize:20,marginBottom:2,color:"#000"}}>#{i+1}</div>
                                      <div style={{fontSize:12,fontWeight:700,color:col,marginBottom:4}}>{run.label}</div>
                                      <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
                                        {[{k:"MAE",v:run.mae.toFixed(0),u:"MW"},{k:"RMSE",v:run.rmse.toFixed(0),u:"MW"},{k:"MAPE",v:run.mape.toFixed(2),u:"%"}].map(m=>(
                                          <div key={m.k} style={{background:"transparent",borderRadius:6,padding:"3px 7px",fontSize:10}}><span style={{color:"#000"}}>{m.k} </span><span style={{color:"#000",fontWeight:600}}>{m.v}{m.u}</span></div>
                                        ))}
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                              </div>
                            </Card>
                            {[{title:"Classement MAE",key:"byMAE",metric:"mae",unit:"MW",color:"#4c72b8"},{title:"Classement RMSE",key:"byRMSE",metric:"rmse",unit:"MW",color:"#2f8d73"},{title:"Classement MAPE",key:"byMAPE",metric:"mape",unit:"%",color:"#c69427"}].map(({title,key,metric,unit,color})=>(
                              <Card key={key} title={title} onExport={()=>csvDownload(rankings[key].map((r,i)=>({rank:i+1,label:r.label,[metric]:r[metric].toFixed(metric==="mape"?2:0)})),`classement_${key}.csv`)}>
                                <div style={{background:"#fff",borderRadius:8,padding:"8px"}}>
                                <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                                  <thead><tr style={{borderBottom:`1px solid ${THEME.border}`}}>{["Rang","MoE",metric.toUpperCase()].map(h=><th key={h} style={{textAlign:"left",padding:"5px 8px",color:THEME.textMuted,fontWeight:600}}>{h}</th>)}</tr></thead>
                                  <tbody>
                                    {rankings[key].map((run,i)=>{
                                      const col=MOE_PALETTE[allRuns.indexOf(run)%MOE_PALETTE.length];
                                      const val=metric==="mape"?run.mape.toFixed(2):run[metric].toFixed(0);
                                      const pct=Math.max(10,(run[metric]/Math.max(...rankings[key].map(r=>r[metric])))*100);
                                      return(
                                        <tr key={run.id} style={{borderBottom:`1px solid ${THEME.border}`}}>
                                          <td style={{padding:"6px 8px",fontWeight:700,color:medalColor(i+1),fontSize:13}}>#{i+1}</td>
                                          <td style={{padding:"6px 8px"}}>
                                            <div style={{display:"flex",alignItems:"center",gap:6}}><span style={{width:8,height:8,borderRadius:"50%",background:col,display:"inline-block"}}/><span style={{color:col,fontWeight:600}}>{run.label}</span></div>
                                            <div style={{marginTop:3,height:3,borderRadius:2,background:THEME.panelBg,overflow:"hidden"}}><div style={{width:`${pct}%`,height:"100%",background:color,borderRadius:2}}/></div>
                                          </td>
                                          <td style={{padding:"6px 8px",textAlign:"right",fontWeight:i===0?700:400,color:i===0?color:THEME.textSecondary}}>{val}{unit}</td>
                                        </tr>
                                      );
                                    })}
                                  </tbody>
                                </table>
                                </div>
                              </Card>
                            ))}
                          </div>
                        )}
                      </>
                    )}
                  </>
                )
              }
            </>
          )}
        </div>
      </div>
    </div>
  );
}
