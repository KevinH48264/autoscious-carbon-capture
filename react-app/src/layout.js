import { forceSimulation, forceManyBody, forceCenter, forceCollide, forceLink } from 'd3-force';

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

export function computeLayout(paperNodes, edgesData, leafClusters, centroidNodes) {
  // Create dummy 'center' nodes and add them to paperNodes first
  centroidNodes.forEach(cluster => {
      // Create dummy 'center' node
      let centerNode = {
        paperId: "center_" + cluster.cluster_id, // Give it a unique id
        citationCount: 0, // Dummy value, not used for anything
        x: cluster.x,
        y: cluster.y,
        classification_id: cluster.classification_id,
        };
      paperNodes.push(centerNode);
  });

  // center everything?
  // paperNodes.forEach(node => {
  //   node.x = 0
  //   node.y = 0
  // })

  let links = [];
  let nodeById = new Map(paperNodes.map(node => [node.paperId, node]));

  // Create links from 1) each paper to their most 2 similar paperIds and 2) MST for each cluster
  edgesData.forEach(edge => {
      if (paperNodes.some(node => node.paperId === edge.source) && paperNodes.some(node => node.paperId === edge.target)) {
        links.push({
            source: nodeById.get(edge.source),
            target: nodeById.get(edge.target),
            strength: 1,
            distance: 10,
        });
      } else {
        console.log("Edge not found in paperNodes:", edge)
      }
  });

  // Create dummy 'center' nodes and links to the strongest paper for each leafCluster
  // let idCounter = edgesData.length;
  // leafClusters.forEach(cluster => {
  //   // Link all nodes in the cluster to the 'center' node
  //     if (cluster.papers.length !== 0) {
  //       const closestPaper = cluster.papers[0]
  //         links.push({
  //           source: closestPaper,
  //           target: "center_" + cluster.id,
  //           value: 10000 // Large weight to keep them close
  //         });

  //         // Add to edgesData for visualization
  //         const newEdge = {
  //           index: idCounter++,
  //           source: closestPaper,
  //           target: "center_" + cluster.id,
  //           value: Math.sqrt(10) // Large weight to keep them close
  //         };
  //         edgesData.push(newEdge); // Add the new edge to edgesData
  //     }

  //     // Link child cluster to parent cluster
  //     const CLUSTER_WEIGHT = 1
  //     const CLUSTER_DISTANCE = 5000
  //     if (cluster.layer - 1 >= 0) {
  //       const parentId = cluster.parents[cluster.layer - 1]
        
  //       links.push({
  //         source: "center_" + parentId,
  //         target: "center_" + cluster.id,
  //         weight: CLUSTER_WEIGHT, // Large weight to keep them close
  //         // distance: CLUSTER_DISTANCE
  //       });

  //       // Add to edgesData for visualization
  //       const newEdge = {
  //         id: idCounter++,
  //         source: "center_" + parentId,
  //         target: "center_" + cluster.id,
  //         weight: Math.sqrt(10), // Large weight to keep them close
  //         distance: 50
  //       };
  //       edgesData.push(newEdge); // Add the new edge to edgesData
  //     } else if (cluster.layer - 1 === -1 && cluster.id !== 0) {
  //       // Link all top nodes to the center node (0)
  //       const parentId = 0
        
  //       links.push({
  //         source: "center_" + parentId,
  //         target: "center_" + cluster.id,
  //         weight: CLUSTER_WEIGHT, // Large weight to keep them close
  //         // distance: CLUSTER_DISTANCE
  //       });

  //       // Add to edgesData for visualization
  //       // const newEdge = {
  //       //   id: idCounter++,
  //       //   source: "center_" + parentId,
  //       //   target: "center_" + cluster.id,
  //       //   weight: Math.sqrt(10) // Large weight to keep them close
  //       // };
  //       // edgesData.push(newEdge); // Add the new edge to edgesData // don't visualize for now so you can see the clusters more easily
  //     }


  //     // Update the center force to the centroid of the current cluster
  //     // simulation.force("center", forceCenter(cluster.centroid_x, cluster.centroid_y));
  //   // }    
  // });

  // Set links

  console.log("SIMULATION PAPER NODES", paperNodes, "LINKS", links)
  let simulation = forceSimulation(paperNodes)
    .force("charge", forceManyBody().strength(-50))
    .force("link", forceLink().links(links).strength(d => d.strength).distance(d => d.distance))
    .force("center", forceCenter(0, 0))
    .stop();


  // Manually iterate the simulation
  let normalizedRadius = 0;
  for (var i = 0; i < 250; ++i) {
    simulation.tick();
    normalizedRadius = normalizeDensity(paperNodes, 0.0007);
  }

  // Remove the dummy 'center' nodes before returning
  let centerNodes = paperNodes.filter(node => node.paperId.startsWith("center_"));
  paperNodes = paperNodes.filter(node => !node.paperId.startsWith("center_"));

  // Now paperNodes have their 'final' position computed by d3 force layout
  return {paperNodes, centerNodes, normalizedRadius, edgesData};
}

// export function computeLayout2(paperNodes, edgesData, clusterData) {

//   return {paperNodes}
// }