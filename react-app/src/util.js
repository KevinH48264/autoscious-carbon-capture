import { Rectangle } from 'pixi.js';
import chroma from 'chroma-js';

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
// export function getLeafClusters(clusterData) {
//   let clusterNodes = [];

//   function recurse(cluster, parentsMap) {
//       const {id, layer, classification_id, name, clusters, papers} = cluster;

//       if (clusters.length > 0 && typeof clusters[0] === 'object') {
//           parentsMap = new Map(parentsMap); // Create a new map at each layer so it doesn't get overridden in sibling branches
//           parentsMap.set(layer, id); // Map the current layer to the current cluster_id
//           clusters.forEach((child) => recurse(child, parentsMap));
//       } else {
//           parentsMap.set(layer, id); // Map the current layer to the current cluster_id
//           const parents = {};
//           for(let [key, value] of parentsMap.entries()) {
//             parents[key] = value;
//           }
//           clusterNodes.push({id, layer, classification_id, clusters, parents, name, papers});
//       }
//   }

//   clusterData.forEach((cluster) => recurse(cluster, new Map()));

//   return clusterNodes;
// }

// Flatten hierarchy to return all clusters with parent cluster ids
export function flattenClusters(clusterData) {
  let clusterNodes = [];

  function recurse(cluster, parentsMap) {
      const {id, layer, classification_id, name, children} = cluster;

      // Create a new map at each layer so it doesn't get overridden in sibling branches
      parentsMap = new Map(parentsMap);
      // Map the current layer to the current cluster_id
      parentsMap.set(layer, id);

      // Create the parents object
      const parents = {};
      for(let [key, value] of parentsMap.entries()) {
          parents[key] = value;
      }

      let clusterPapers = []
      children.forEach((child) => {
        if (child.value) {
          clusterPapers.push(child.value)
        }
      })


      // Push the current cluster to clusterNodes
      clusterNodes.push({id, layer, classification_id, children, parents, name, papers: clusterPapers});

      // If the cluster has child clusters, recurse on them
      children.forEach((child) => {
        if (child.id) {
          recurse(child, parentsMap)
        }
      })
  }

  clusterData[0].children.forEach((cluster) => recurse(cluster, new Map()));

  return clusterNodes;
}

  

  // Assigning Voronoi points to clusters
  // export function flattenClusters(clusterData, voronoi, leafClusters) {
  //   let clusterNodes = [];

  //   function recurse(cluster) {
  //     const {cluster_id, layer, centroid_x, centroid_y, content} = cluster;
  //     const leafClusterIndex = leafClusters.findIndex(leafCluster => leafCluster.cluster_id === cluster_id);

  //     // if it's a leaf cluster with Voronoi points
  //     if (leafClusterIndex !== -1) {
  //       const cellPolygon = voronoi.cellPolygon(leafClusterIndex);

  //       if (cellPolygon) {
  //         cluster.voronoiPoints = cellPolygon;
  //       } else {
  //         cluster.voronoiPoints = [];
  //       }
  //     }

  //     // otherwise it's a parent cluster
  //     else {
  //       cluster.content.forEach(recurse);

  //       // If it's a parent cluster, gather Voronoi points from child clusters
  //       let allPoints = cluster.content.flatMap(child => child.voronoiPoints);

  //       if (allPoints[0] === []) {
  //         cluster.voronoiPoints = []
  //       } else {
  //         let centerX = allPoints.reduce((sum, point) => sum + point[0], 0) / allPoints.length;
  //         let centerY = allPoints.reduce((sum, point) => sum + point[1], 0) / allPoints.length;

  //         cluster.voronoiPoints = sortPoints(allPoints, centerX, centerY);
  //       }
  //     } 

  //     clusterNodes.push({cluster_id, layer, centroid_x, centroid_y, content, voronoiPoints: cluster.voronoiPoints});
  //   }

  //   clusterData.forEach(recurse);

  //   return clusterNodes;
  // }
  
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

// Function to map cluster id to main topic
export const traverseCluster = (cluster, clusterMap) => {
  clusterMap.set(cluster.id, cluster.name);

  // Check if this cluster has child clusters.
    cluster.children.forEach(childCluster => {
      // only clusters have ids, papers don't have id, they only have name and value
      if (childCluster.id !== undefined) {
        traverseCluster(childCluster, clusterMap)
      }
    });
}

export const calculateClusterCentroids = (leafClusters, paperNodes, centroidNodes) => {
  leafClusters.forEach(cluster => {
    // TODO: Actually just take the centroid for topics of everything.

    // Calculate and set the topic for this cluster
    let topicNode = paperNodes.find(node => node.data.classification_id === cluster.classification_id)

    // Link all nodes in the cluster to the 'center' node
    let sumX = topicNode.x;
    let sumY = topicNode.y;
    let count = 1;
  
    cluster.papers.forEach(paperId => {
      // Find the paperNode with the same paperId
      let paperNode = paperNodes.find(node => node.data.value === paperId);
      if (paperNode) {
        sumX += paperNode.x;
        sumY += paperNode.y;
        count++;
      }
    });
  
    
    let centroidX = count > 0 ? sumX / count : 0;
    let centroidY = count > 0 ? sumY / count : 0;
  
    cluster.centroid_x = centroidX;
    cluster.centroid_y = centroidY;

    centroidNodes.push({x: centroidX, y: centroidY, cluster_id: cluster.id, classification_id: cluster.classification_id, name: cluster.name})
  });    
}

export function getVoronoiNeighbors(voronoi, i) {
  return Array.from(voronoi.neighbors(i));
}  

// Diamond colors
export const diamondPolygonColorSequence = [
  0xB9F2FF, // Very Pale Blue
  0xA4C8F0, // Light Sky Blue
  0x89A1D0, // Periwinkle Blue
  0x6D8CB3, // Cadet Blue
  0x507696, // Steel Blue
  0x335B7A, // Dark Slate Blue
  0x16405F, // Dark Imperial Blue
  0xD8F2E1, // Very Pale Green
  0xB5E0D3, // Light Cyan
  0x92CEC5  // Medium Aquamarine
];

// const oldBaseColors = [
//   "rgb(0, 0, 139)",    // DarkBlue
//   "rgb(139, 0, 0)",    // DarkRed
//   "rgb(0, 139, 139)",  // DarkCyan
//   "rgb(107, 142, 35)", // OliveDrab
//   "rgb(34, 139, 34)",   // ForestGreen
//   "rgb(139, 0, 139)",  // DarkMagenta
//   "rgb(47, 79, 79)",   // DarkSlateGray
//   "rgb(160, 82, 45)",  // Sienna
//   "rgb(112, 128, 144)", // SlateGray
//   "rgb(0, 0, 0)",      // Black
// ];

const baseColors = [
  "rgb(0, 0, 139)",     // DarkBlue
  "rgb(0, 0, 205)",     // MediumBlue
  "rgb(139, 0, 0)",     // DarkRed
  "rgb(255, 140, 0)",   // DarkOrange
  "rgb(0, 139, 139)",   // DarkCyan
  "rgb(0, 205, 205)",   // MediumTurquoise
  "rgb(107, 142, 35)",  // OliveDrab
  "rgb(34, 139, 34)",   // ForestGreen
  "rgb(105, 105, 105)", // DimGray
  "rgb(0, 0, 0)",       // Black
];

const colorMap = {};

export const getColorForClass = (classId) => {
  // console.log("Class id: ", classId)
  const parts = classId.split(".");
  const topLevelIndex = parseInt(parts[0], 10) - 1;
  const colorIndex = topLevelIndex % baseColors.length;  // use modulus to wrap around

  // If we haven't seen this top level before, assign it a base color and create a scale
  if (!colorMap[parts[0]]) {
      colorMap[parts[0]] = {
          base: chroma(baseColors[colorIndex]).hex(),
          scale: chroma.scale([chroma(baseColors[colorIndex]).hex(), 'white']).mode('lch').colors(20)
      };
  }

  // Now get a specific hue for the subtopic based on the subcategory index
  let color = colorMap[parts[0]].scale[0];
  for (let i = 1; i < parts.length; i++) {
      const subcategoryIndex = parseInt(parts[i], 10);
      color = chroma.mix(color, colorMap[parts[0]].scale[subcategoryIndex], 0.5, 'lch').hex();
  }
  
  return color;
};




// export const assignHues = (nodes, min_hue, max_hue) => {
//     const range = max_hue - min_hue;
//     let sorted = nodes.sort((a, b) => this.Cluster_Nodes[a.id].length - this.Cluster_Nodes[b.id].length);
//     // shuffle the last n-2 elements
//     let shuffled = _.shuffle(sorted.slice(2));
//     shuffled = sorted.slice(0, 2).concat(shuffled);
//     // swap the second element with the middle element
//     let middle = Math.floor(shuffled.length / 2);
//     [shuffled[1], shuffled[middle]] = [shuffled[middle], shuffled[1]];

//     for (let i = 0; i < shuffled.length; i++) {
//         shuffled[i].hue = range * i / shuffled.length + min_hue;
//     }
// }

// export const circleColorDiamondSequence = [
//   0xB2DDFF, // Pale Blue
//   0xFFC1D6, // Soft Pink
//   0xD3D3D3, // Light Grey
//   0xFFD700, // Gold
//   0xA0C1D1, // Sky Blue
//   0xFCE4EC, // Pinkish White
//   0x9FA8DA, // Cool Greyish Blue
//   0xE6E2D3, // Very Light Yellowish Grey
//   0xC5E1A5, // Soft Green
//   0xBCAAA4, // Earthy Grey
//   0x80DEEA, // Light Cyan
//   0xF48FB1, // Light Pink
// ];




              // Visualizing the bounds
              // if (node.graphics) {
              //   viewport.removeChild(node.graphics);
              // }
              // node.graphics = new PIXI.Graphics();
              // node.graphics.lineStyle(1, 0xFF0000, 1); // Set line style (width, color, alpha)
              // node.graphics.drawRect(-node_bound.width / 2, -node_bound.height / 10, node_bound.width, node_bound.height); 
              // node.graphics.position.set(node.x + circleHeight, node.y + circleHeight);

              // viewport.addChild(node.graphics); 
          // }

      // Visualizing centroid nodes from force directed simulation
      // layout.centerNodes.forEach((node, i) => {  
      //   // Handling Node text, draw labels
      //   const debug_factor = 1
      //   const lambda = debug_factor
      //   // const lambda = debug_factor * (Math.sqrt(node.citationCount) - min_scale) / (max_scale - min_scale);
      //   const fontSize = min_font_size + (max_font_size - min_font_size) * lambda;
      //   const circleHeight = 1 + 4 * lambda;

      //   if(!node.circle) {
      //       node.circle = new PIXI.Graphics();
      //       node.circle.beginFill(0xb9f2ff);
      //       node.circle.drawCircle(node.x, node.y, circleHeight);
      //       node.circle.endFill();
      //       viewport.addChild(node.circle);
      //   } else {
      //       node.circle.visible = true; // make it visible if it already exists
      //   }
      // })

      // Checking for taxonomy code
        // Using the node.paperId, find the topic and subtopic in the taxonomy
        // for (let topic in taxonomy) {
        //   for (let subtopic in taxonomy[topic]) {
        //     if (taxonomy[topic][subtopic].includes(node.paperId)) {
        //       node.topic = topic;
        //       node.subtopic = subtopic;
        //       break;
        //     }
        //   }
        // }
        
        // if (!node.topic || !node.subtopic) {
        //   // If the paperId is not in the taxonomy, assign a default value or handle it differently
        //   node.topic = "Unknown";
        //   node.subtopic = "Unknown";
        //   console.log("UNKNOWN", node.paperId, node.title, node.topic, node.subtopic)
        // }

// Code for removing polygons from container
// const addClusterPolygons = (node, i, opacity, currentLevel) => {
//   // You want the polygon of the parents, not the current node which would be node.parents[currentLevel]
//   let parentId = node.parents[currentLevel + 1];
//   if (parentId === undefined) {
//     parentId = node.parents[Math.max(...Object.keys(node.parents).map(Number))];
//   }
//   let parentClassId = clusterToClassId.get(parentId)
//   let fillColor = classColorMap.get(parentClassId);

//   const region = scaledVoronoi[i];
//   const polygon = new PIXI.Graphics();
//   polygon.zIndex = 50;

//   polygon.beginFill(fillColor, opacity * 0.7);
//   polygon.drawPolygon(region.map(([x, y]) => new PIXI.Point(x, y)));
//   polygon.endFill();

//   node.region = polygon;
//   polygonContainer.addChild(polygon);
// }

// leafClusters.forEach((node, i) => {
//   addClusterPolygons(node, i, 1 - zoomDecimalToNextZoom, zoomLevel)
//   addClusterPolygons(node, i, zoomDecimalToNextZoom, zoomLevel + 1)
// });

// Manually inspecting edges between topics
// let nodeById = new Map(layoutNodes.map(node => [node.data.paperId, node]));
      // let edgeTempLinks = []
      // edgesData.forEach(edge => {
      //   edgeTempLinks.push({
      //           source: nodeById.get(edge.source),
      //           target: nodeById.get(edge.target),
      //           strength: 0,
      //           distance: 0,
      //           weight: edge.weight,
      //       });
      //   });
      // console.log("nodeById", nodeById, "layoutNodes", layoutNodes)
      // edgeTempLinks.forEach(edge => {
      //   // optimize this later!
      //     // console.log(edge.source)
      //     const sourceNode = layoutNodes.find(node => node === edge.source);
      //     const targetNode = layoutNodes.find(node => node === edge.target);

      //     let edgeColor = 0x808080
      //     let edgeOpacity = 1

      
      //     // Create a new graphics object for the edge if it doesn't exist
      //     if (!edge.edge_graphics) {
      //       edge.edge_graphics = new PIXI.Graphics();
      //       edge.edge_graphics.zIndex = 50; // set this below node's zIndex to ensure nodes are drawn on top
      //       viewport.addChild(edge.edge_graphics);

      //       // Draw the line
      //       edge.edge_graphics.clear(); // remove any existing line
      //       edge.edge_graphics.lineStyle(1, edgeColor, edgeOpacity); 
      //       edge.edge_graphics.moveTo(sourceNode.x, sourceNode.y); // move to the source node's position
      //       edge.edge_graphics.lineTo(targetNode.x, targetNode.y); // draw a line to the target node's position
      //       viewport.addChild(edge.edge_graphics)
      //     } else {
      //       edge.edge_graphics.visible = true;  
      //     }
      // });