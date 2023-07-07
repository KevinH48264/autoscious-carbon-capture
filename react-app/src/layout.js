import { forceSimulation, forceManyBody, forceCenter, forceCollide, forceLink } from 'd3-force';

export function computeLayout(paperNodes, edgesData, leafClusters) {
    // Create dummy 'center' nodes and add them to paperNodes first
    leafClusters.forEach(cluster => {
        // Create dummy 'center' node
        let centerNode = {
        paperId: "center_" + cluster.cluster_id, // Give it a unique id
        citationCount: 0 // Dummy value, not used for anything
        };
        paperNodes.push(centerNode);
    });

  let simulation = forceSimulation()
    .nodes(paperNodes) // Set nodes
    .force("charge", forceManyBody().strength(-100))
    .force("link", forceLink().id(d => d.paperId)) // This is how we access the id of each node
    .force("center", forceCenter(0, 0)) // Initial center, this will be updated for each leaf cluster
    .force("collision", forceCollide().radius(function(d) {
        // TODO: this is usually determined by the radius, update radius if not citationCount!
      return Math.sqrt(d.citationCount);
    }))
    .stop();

  let links = [];

  // Create links from EdgesData
  edgesData.forEach(edge => {
    if (!paperNodes.find(node => node.paperId === edge.source)) {
        console.error(`Node not found: ${edge.source}`);
      }
      if (!paperNodes.find(node => node.paperId === edge.target)) {
        console.error(`Node not found: ${edge.target}`);
      }

    links.push({
      source: edge.source,
      target: edge.target,
      weight: edge.weight
    });
  });

  // Create dummy 'center' nodes and links to them for each leafCluster
  leafClusters.forEach(cluster => {
    // Link all nodes in the cluster to the 'center' node
    cluster.content.forEach(paperId => {
      links.push({
        source: paperId,
        target: "center_" + cluster.cluster_id,
        weight: 1000 // Large weight to keep them close
      });
    });

    // Update the center force to the centroid of the current cluster
    simulation.force("center", forceCenter(cluster.centroid_x, cluster.centroid_y));
  });


  // Set links
  simulation.force("link").links(links);

  // Manually iterate the simulation
  for (var i = 0; i < 300; ++i) simulation.tick();

  // Remove the dummy 'center' nodes before returning
  paperNodes = paperNodes.filter(node => !node.paperId.startsWith("center_"));

  console.log("returning papernodes after finishing simulation!")

  // Now paperNodes have their 'final' position computed by d3 force layout
  return paperNodes;
}
