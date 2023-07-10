import { Rectangle } from 'pixi.js';

export const randomDarkModeColor = () => {
    // vary hue between 0 and 360 degrees
    const hue = Math.random() * 360;
    // keep saturation at 80% for rich colors
    const saturation = 80;
    // vary lightness between 30 and 70% to keep colors neither too dark nor too bright
    const lightness = 30 + Math.random() * 40;
    
    return `hsla(${hue},${saturation}%,${lightness}%, 1)`;
}

export const rectIntersectsRect = (a, b) => {
    if (a.x > b.x + b.width || a.x + a.width < b.x)
        return false;
    if (a.y > b.y + b.height || a.y + a.height < b.y)
        return false;
    return true;
}

// Sort points in counterclockwise order
export const sortPoints = (points, centerX, centerY) => {
    return points.sort((a, b) => {
    return Math.atan2(a[1] - centerY, a[0] - centerX) - Math.atan2(b[1] - centerY, b[0] - centerX);
    });
}

// creating clusters
export function getLeafClusters(clusterData) {
    let clusterNodes = [];
  
    function recurse(cluster, parentsMap) {
      cluster.voronoiPoints = [];

      const {cluster_id, layer, centroid_x, centroid_y, polygonPoints, content} = cluster;

      if (typeof cluster.content[0] === 'object' && cluster.content[0] !== null) {
        parentsMap = new Map(parentsMap); // Create a new map at each layer so it doesn't get overridden in sibling branches
        parentsMap.set(layer, cluster_id); // Map the current layer to the current cluster_id
        cluster.content.forEach((child) => recurse(child, parentsMap));
      } else {
        parentsMap.set(layer, cluster_id); // Map the current layer to the current cluster_id
        const parents = {};
        for(let [key, value] of parentsMap.entries()) {
          parents[key] = value;
        }
        clusterNodes.push({cluster_id, layer, centroid_x, centroid_y, content, polygonPoints, parents});
      }
    }
  
    clusterData.forEach((cluster) => recurse(cluster, new Map()));
  
    return clusterNodes;
}
  

  // Assigning Voronoi points to clusters
  export function flattenClusters(clusterData, voronoi, leafClusters) {
    let clusterNodes = [];

    function recurse(cluster) {
      const {cluster_id, layer, centroid_x, centroid_y, content} = cluster;
      const leafClusterIndex = leafClusters.findIndex(leafCluster => leafCluster.cluster_id === cluster_id);

      // if it's a leaf cluster with Voronoi points
      if (leafClusterIndex !== -1) {
        const cellPolygon = voronoi.cellPolygon(leafClusterIndex);

        if (cellPolygon) {
          cluster.voronoiPoints = cellPolygon;
        } else {
          cluster.voronoiPoints = [];
        }
      }

      // otherwise it's a parent cluster
      else {
        cluster.content.forEach(recurse);

        // If it's a parent cluster, gather Voronoi points from child clusters
        let allPoints = cluster.content.flatMap(child => child.voronoiPoints);

        if (allPoints[0] === []) {
          cluster.voronoiPoints = []
        } else {
          let centerX = allPoints.reduce((sum, point) => sum + point[0], 0) / allPoints.length;
          let centerY = allPoints.reduce((sum, point) => sum + point[1], 0) / allPoints.length;

          cluster.voronoiPoints = sortPoints(allPoints, centerX, centerY);
        }
      } 

      clusterNodes.push({cluster_id, layer, centroid_x, centroid_y, content, voronoiPoints: cluster.voronoiPoints});
    }

    clusterData.forEach(recurse);

    return clusterNodes;
  }
  
export const multilineText = (text, charLength) => {
  let words = text.split(' ');
  let lines = [];
  let currentLine = '';
  
  for (let word of words) {
      if ((currentLine + ' ' + word).length > charLength) {
          lines.push(currentLine);
          currentLine = word;
      } else {
          currentLine += ' ' + word;
      }
  }
  lines.push(currentLine);
  
  let multilineTitle = lines.join('\n').trim();
  return multilineTitle
}

export const labelBounds = (font_size, x_in, y_in, text_length=30, multilineTitle="") => {
  // approximate the bounding box of the label. getBounds is slow

  const num_new_lines = multilineTitle.split("\n").length;
  const scale_x = 0.5;
  const scale_y = 1.5;

  let height = font_size * num_new_lines * scale_y;
  let width = multilineTitle.length;
  if (multilineTitle.length > text_length) {
    width = text_length;
  }
  width = width * font_size * scale_x;
  
  // Look at anchor, (0.5, 0.5) requires (- width / 2, - height / 2)
  const x = x_in - width / 2;
  const y = y_in - height / 10;
  const bounds = new Rectangle(x, y, width, height);
  return bounds;
}

// Define the start (diamond) and end (white) colors in RGB
let startColor = {r: 0, g: 191, b: 255};
let endColor = {r: 255, g: 255, b: 255};

// Function to calculate the color at a specific zoom level
export function getColorAtZoomLevel(zoomLevel, zoomLayers) {
    let r = Math.round(startColor.r + (endColor.r - startColor.r) * (zoomLevel / zoomLayers));
    let g = Math.round(startColor.g + (endColor.g - startColor.g) * (zoomLevel / zoomLayers));
    let b = Math.round(startColor.b + (endColor.b - startColor.b) * (zoomLevel / zoomLayers));
  
    // Convert the RGB value to a hexadecimal color code
    let hexColor = ((r << 16) | (g << 8) | b).toString(16);
  
    // Pad with leading zeros, if necessary
    while (hexColor.length < 6) {
      hexColor = "0" + hexColor;
    }
  
    return "#" + hexColor;
}

// Random colors
// export const colorSequence = [
//     14413679,
//     8130704,
//     11765267,
//     2811802,
//     1221959,
//     15148973,
//     2025023,
//     1290288,
//     5957152,
//     15195948,
//     13494055,
//     13047398,
//     2453478,
//     13298244,
//     10874478,
//     1477575,
//     8952337,
//     4711914,
//     11360531,
//     2233264,
//     13355030,
//     10606616,
//     6199825,
//     3120657,
//     5115786,
//     4254027,
//     15150256,
//     15489713,
//     3401827,
//     3532992,
//     10455825,
//     14424184,
//     15383624,
//     15300926,
//     1066133,
//     15556767,
//     6679111,
//     10517009,
//     14108439,
//     13756737,
//     7112174,
//     6942351,
//     15232560,
//     13506111,
//     1405378,
//     9113438,
//     1472461,
//     9375616,
//     13703959,
//     15148865,
//     2278118,
//     10687100,
//     1290629,
//     1288997,
//     11867853,
//     13691458,
//     2297743,
//     6730478,
//     14151536,
//     12447073,
//     10546267,
//     1329076,
//     9916432,
//     6469396,
//     3598565,
//     4174057,
//     1396417,
//     15498584,
//     1659873,
//     11735978,
//     9170799,
//     15285888,
//     1535703,
//     12840025,
//     13198102,
//     6654735,
//     7007908,
//     1019196,
//     15709807,
//     12175638,
//     5941740,
//     1324982,
//     15197485,
//     14008599,
//     6427305,
//     3527447,
//     10162574,
//     12207338,
//     5059560,
//     13049830,
//     5499730,
//     14525720,
//     11870950,
//     15150511,
//     6084076,
//     2221714,
//     1153554,
//     9049275,
//     15444303,
//     11817707,
//     12840285,
//     7181294,
//     1087345,
//     11408147,
//     1018402,
//     13671959,
//     15625099,
//     11094250,
//     2429073,
//     2352849,
//     2238694,
//     15421518,
//     13572313,
//     15372874,
//     12391711,
//     1351608,
//     14424173,
//     7048207,
//     10490220,
//     15168295,
//     15043867,
//     13888336,
//     11425811,
//     2221681,
//     5968287,
//     15693002,
//     10131217,
//     15422050,
//     2483868,
//     5332715,
//     15692402,
//     11868379,
//     2910951,
//     3269915,
//     3055889,
//     15365961,
//     8874478,
//     9284371,
//     6061548,
//     4043497,
//     7410607,
//     9900442,
//     1085486,
//     15081094,
//     5106575,
//     4307177,
//     1087051,
//     3467463,
//     8688912,
//     1356956,
//     4582100,
//     12538347,
//     8162575,
//     15427919,
//     1403071,
//     15238454,
//     15588453,
//     5485844,
//     15150169,
//     6089921,
//     11080284,
//     14565912,
//     15150471,
//     12505111,
//     5630188,
//     4426255,
//     15694448,
//     4170217,
//     4123046,
//     2680690,
//     5696727,
//     1345204,
//     11261976,
//     9345553,
//     3171816,
//     13917207,
//     6152429,
//     9411601,
//     6733294,
//     7375631,
//     13261334,
//     15218901,
//     5827778,
//     2723856,
//     8187236,
//     8318566,
//     1415877,
//     15589474,
//     15111206,
//     1493950,
//     12380212,
//     8720269,
//     4758545,
//     1493029,
//     4527556,
//     8434195,
//     15217096,
//     15065114,
//     10162585,
//     10656530,
//     6679378,
//     3806657,
//     13195500,
//     10687091,
//     14725656,
//     4647583,
//     15691904,
//     15150409,
//     15556519,
//     6691033,
//     15149890,
//     7334470,
//     2465809,
//     1331125,
//     4332261,
//     11661170,
//     15692665,
//     12327133,
//     11539378,
//     15285636,
//     15378502,
//     1289108,
//     2615252,
//     15286600,
//     14412356,
//     1426821,
//     1129120,
//     8852667,
//     4188505,
//     8394974,
//     3205181,
//     1425503,
//     1425497,
//     15712369,
//     4199077,
//     15186219,
//     10217259,
//     11152615,
//     3412910,
//     1630140,
//     1515219,
//     7466910,
//     15285165,
//     7902480,
//     14817747,
//     12460821,
//     15357258,
//     5093396,
//     1330105,
//     11604891,
//     15353533,
//     11195927,
//     7597807,
//     1290339,
//     13375126,
//     10424994,
//     15285634,
//     14358690,
//     10950077,
//     8460243,
//     15431250,
//     15625594,
//     15420512,
//     7378159,
//     1493783,
//     1489612,
//     6217453,
//     2103250,
//     1186218,
//     2693336,
//     7479014,
//     2004752,
//     3008414,
//     5892148,
//     7532469,
//     12391859,
//     15436364,
//     15217769,
//     1124257,
//     6427832,
//     10163122,
//     5510329,
//     10252014,
//     13560124,
//     9572469,
//     7611366,
//     14740035,
//     2876966,
//     5574538,
//     10774290,
//     11801783,
//     12063919,
//     1511569,
//     1355961,
//     12040980,
//     15420649,
//     6674925,
//     4353514,
//     8132030,
//     1358226,
//     1696934
// ]

// Diamond colors
export const colorSequence = [
  0xffffff,
  0xf8f8f8,
  0xf1f1f1,
  0xeaeaea,
  0xe3e3e3,
  0xdcdcdc,
  0xd5d5d5,
  0xcecece,
  0xc7c7c7,
  0xc0c0c0,
  0xb9b9b9,
  0xb2b2b2,
  0xababab,
  0xa4a4a4,
  0x9d9d9d,
  0x969696,
  0x8f8f8f,
  0x888888,
  0x818181,
  0x7a7a7a,
  0x737373,
  0x6c6c6c,
  0x656565,
  0x5e5e5e,
  0x575757,
  0x505050,
  0x494949,
  0x424242,
  0x3b3b3b,
  0x343434,
  0x2d2d2d,
  0x262626,
  0x1f1f1f,
  0x181818,
  0x111111,
  0x0a0a0a,
  0x030303,
  0x0a0a0a,
  0x111111,
  0x181818,
  0x1f1f1f,
  0x262626,
  0x2d2d2d,
  0x343434,
  0x3b3b3b,
  0x424242,
  0x494949,
  0x505050,
  0x575757,
  0x5e5e5e,
  0x656565,
  0x6c6c6c,
  0x737373,
  0x7a7a7a,
  0x818181
];
