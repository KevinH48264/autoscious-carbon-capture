import { forceSimulation, forceManyBody, forceCenter, forceCollide, forceLink, forceX, forceY } from 'd3-force';
import { hierarchy } from 'd3';

function normalizeDensity(nodes, target_density = 0.0007, x0=0, y0=0) {
  const max_norm = Math.max(...nodes.map(node => Math.sqrt(node.x * node.x + node.y * node.y)));
  const area = Math.PI * max_norm * max_norm;
  const target_area = nodes.length / target_density;
  const norm_scale = Math.sqrt(target_area / area);
  for (let node of nodes) {
      node.x *= norm_scale;
      node.y *= norm_scale;

      node.x += x0;
      node.y += y0;
  }

  return max_norm * norm_scale;
}


export function computeHierarchicalLayout(clusterData, paperNodes, edgesData) {
  // Completely based on taxonomy_json, so extra papers not clasified in taxonomy json won't appear
  const root = hierarchy(clusterData[0])
  let links = root.links();
  const nodes = root.descendants() // uses a name value where name = id, value is a list of attributes

  console.log("layout debug", "nodes: ", nodes, "paperNodes: ", paperNodes)

  // paper nodes map for faster lookup
  let paperNodesMap = new Map();
  paperNodes.forEach(paperNode => paperNodesMap.set(paperNode.paperId, paperNode));

  nodes.forEach(node => {
    if (node.data.value) {
      // TODO: optimize this, this is probably time consuming
      let paperNode = paperNodesMap.get(node.data.value[0].paperId);
      if (paperNode) {
        node.data.abstract = paperNode.abstract;
        node.data.authors = paperNode.authors;
        node.data.citationCount = paperNode.citationCount;
        node.data.citations = paperNode.citations;
        node.data.classification_ids = paperNode.classification_ids;
        node.data.doi = paperNode.doi;
        node.data.isOpenAccess = paperNode.isOpenAccess;
        node.data.language = paperNode.language;
        node.data.publication_date = paperNode.publication_date;
        node.data.relevance_score = paperNode.relevance_score;
        node.data.year = paperNode.year;
        node.data.title = paperNode.title;
        node.data.x = paperNode.x;
        node.data.y = paperNode.y;
        node.data.url = paperNode.url;
        node.data.paperId = paperNode.paperId;
      }
    }
  });

  // Normalize edge weights
  let max_weight = Math.max(...edgesData.map(edge => edge.weight));
  edgesData.forEach(edge => {
      edge.weight /= max_weight;
  });

  // Create links from each paper to their most similar paperIds within the cluster
  let nodeById = new Map(nodes.map(node => [node.data.name, node]));
  edgesData.forEach(edge => {
      links.push({
          source: nodeById.get(edge.source),
          target: nodeById.get(edge.target),
          strength: 0,
          distance: 0,
          weight: edge.weight,
      });
  });


  console.log("SIMULATION NODES: ", nodes, "SIMULATION LINKS", links)
  const simulation = forceSimulation(nodes)
      .force("link", forceLink(links).id(d => d.id).distance(d => d.dstance ? d.distance : 0).strength(d => d.strength ? d.strength : 1.5))
      .force("charge", forceManyBody().strength(d => d['data'].value ? -50 : -10))
      .force("x", forceX())
      .force("y", forceY())
      .force("center", forceCenter(0, 0))
      .stop()

  // Manually iterate the simulation
  let normalizedRadius = 0;
  for (var i = 0; i < 250; ++i) {
    simulation.tick();
    normalizedRadius = normalizeDensity(nodes, 0.007);
  }

  return { nodes, links, normalizedRadius };
}

// export function computeLayout(paperNodes, edgesData, leafClusters, centroidNodes) {
//   // Add dummy 'center' nodes to paperNodes first
//   centroidNodes.forEach(cluster => {
//       // Create dummy 'center' node
//       let centerNode = {
//         paperId: "center_" + cluster.cluster_id, // Give it a unique id
//         citationCount: 0, // Dummy value, not used for anything
//         x: cluster.x,
//         y: cluster.y,
//         classification_id: cluster.classification_id,
//         };
//       paperNodes.push(centerNode);
//   });

//   paperNodes.forEach(node => {
//     node.x = 0
//     node.y = 0
//   })

//   let links = [];
//   let nodeById = new Map(paperNodes.map(node => [node.paperId, node]));

//   // Create links from 1) each paper to their most 2 similar paperIds and 2) MST for each cluster
//   // max strength between paper in same subcluster and reasonable distance for spread
//   edgesData.forEach(edge => {
//       links.push({
//           source: nodeById.get(edge.source),
//           target: nodeById.get(edge.target),
//           strength: 1,
//           distance: 50,
//       });
//   });

//   // Create dummy 'center' nodes and links to the strongest paper for each leafCluster
//   let idCounter = edgesData.length;
//   leafClusters.forEach(cluster => {
//     // Link closest nodes in the cluster to the 'center' node
//       if (cluster.papers.length !== 0) {
//         const closestPaper = cluster.papers[0]
//           links.push({
//             source: nodeById.get(closestPaper),
//             target: nodeById.get("center_" + cluster.id),
//             strength: 2,
//             distance: 0,
//           });

//           // Add to edgesData for visualization
//           const newEdge = {
//             index: idCounter++,
//             source: closestPaper,
//             target: "center_" + cluster.id,
//             value: Math.sqrt(10) // Large weight to keep them close
//           };
//           edgesData.push(newEdge); // Add the new edge to edgesData
//       }

//       // Link child cluster to parent cluster
//       // Low weight / attraction but still there, and high ditance to allow subcluster location flexibility and prevent overlap
//       const CLUSTER_WEIGHT = 0.1
//       const CLUSTER_DISTANCE = 60
//       if (cluster.layer - 1 >= 0) {
//         const parentId = cluster.parents[cluster.layer - 1]
        
//         links.push({
//           source: nodeById.get("center_" + parentId),
//           target: nodeById.get("center_" + cluster.id),
//           strength: CLUSTER_WEIGHT,
//           distance: CLUSTER_DISTANCE,
//         });

//         // Add to edgesData for visualization
//         const newEdge = {
//           id: idCounter++,
//           source: "center_" + parentId,
//           target: "center_" + cluster.id,
//           weight: CLUSTER_WEIGHT, // Large weight to keep them close
//           distance: CLUSTER_DISTANCE
//         };
//         edgesData.push(newEdge); // Add the new edge to edgesData
//       } 
//       else if (cluster.layer - 1 === -1 && cluster.id !== 0) {
//         // Link all top nodes to the center node (0)
//         const parentId = 0
        
//         links.push({
//           source: nodeById.get("center_" + parentId),
//           target: nodeById.get("center_" + cluster.id),
//           strength: CLUSTER_WEIGHT,
//           distance: CLUSTER_DISTANCE,
//         });

//         // Add to edgesData for visualization
//         // const newEdge = {
//         //   id: idCounter++,
//         //   source: "center_" + parentId,
//         //   target: "center_" + cluster.id,
//         //   weight: Math.sqrt(10) // Large weight to keep them close
//         // };
//         // edgesData.push(newEdge); // Add the new edge to edgesData // don't visualize for now so you can see the clusters more easily
//       }
//   });

//   let simulation = forceSimulation(paperNodes)
//     .force("charge", forceManyBody().strength((node) => node.paperId.startsWith("center_") ? -5 : -50))
//     .force("link", forceLink().links(links).strength(d => d.strength).distance(d => d.distance))
//     // .force("center", forceCenter(0, 0))
//     .stop();

//   // Manually iterate the simulation
//   let normalizedRadius = 0;
//   for (var i = 0; i < 300; ++i) {
//     simulation.tick();
//     normalizedRadius = normalizeDensity(paperNodes, 0.0007);
//   }

//   // Remove the dummy 'center' nodes before returning
//   let centerNodes = paperNodes.filter(node => node.paperId.startsWith("center_"));
//   paperNodes = paperNodes.filter(node => !node.paperId.startsWith("center_"));

//   // paperNodes with 'final' position
//   return {paperNodes, centerNodes, normalizedRadius, edgesData};
// }

// There's some fun layout stuff that you can play around with to get the layout showing clockwise, but that's not the focus now
function getMaxClassificationId(clusterData) {
  let maxClassificationId = 0;  // Initialize max to 0

  function helper(node) {
      if (node.classification_id) {
          let floatId = parseFloat(node.classification_id);
          maxClassificationId = Math.max(maxClassificationId, floatId);
      }
      
      // Traverse the "children" array if it exists
      if (Array.isArray(node.children)) {
          for (let child of node.children) {
              helper(child);
          }
      }
  }

  helper(clusterData);

  return maxClassificationId;
}

function assignPolarCoordinates(clusterData, maxClassificationId) {
  const convertToRadians = (angle) => angle * (Math.PI / 180);  // Function to convert degrees to radians

  function helper(node) {
      if (node.classification_id) {
          let floatId = parseFloat(node.classification_id);

          if (floatId < 2) {
            node.polar_x = 0
            node.polar_y = 0
          } else {
            let r = 1.5; // Set the radius to a constant value of 1
            let theta = convertToRadians((360 / maxClassificationId - 2) * floatId - 2);  // Use floatId / maxClassificationId to calculate theta

            node.polar_x = r * Math.cos(theta);  // Convert polar to Cartesian coordinates
            node.polar_y = r * Math.sin(theta);
          }
      }

      // Traverse the "children" array if it exists
      if (Array.isArray(node.children)) {
          for (let child of node.children) {
              helper(child);
          }
      }
  }

  helper(clusterData);
}

function logSortedCoordinates(clusterData) {
  let dataArray = [];

  function helper(node) {
      if (node.classification_id) {
          dataArray.push({
              classification_id: parseFloat(node.classification_id),
              polar_x: node.polar_x,
              polar_y: node.polar_y
          });
      }

      // Traverse the "children" array if it exists
      if (Array.isArray(node.children)) {
          for (let child of node.children) {
              helper(child);
          }
      }
  }

  helper(clusterData);

  // Sort array based on classification_id
  dataArray.sort((a, b) => a.classification_id - b.classification_id);

  // Log each object in sorted order
  for (let data of dataArray) {
      console.log(`id: ${data.classification_id}, x: ${data.polar_x.toFixed(0)}, y: ${data.polar_y.toFixed(0)}`);
  }
}
