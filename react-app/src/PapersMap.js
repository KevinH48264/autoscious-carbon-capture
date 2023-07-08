import React, { useEffect, useRef } from 'react';
import * as PIXI from 'pixi.js';
import { Viewport } from 'pixi-viewport';
import { Delaunay } from 'd3-delaunay';
import { randomDarkModeColor, rectIntersectsRect, sortPoints, getLeafClusters, flattenClusters, colorSequence, multilineText } from './util';
import { computeLayout } from './layout';

const ResearchPaperPlot = ({ papersData, edgesData, clusterData }) => {
  const pixiContainer = useRef();
  PIXI.BitmapFont.from("TitleFont", {
    fill: 0xF8F8F8,
    fontSize: 80,
  }, {
    chars: PIXI.BitmapFont.ASCII.concat(['∀']),
  });
  PIXI.BitmapFont.from("TopicFont", {
    fill: 0xffffff,
    fontSize: 20,
  }, {
    chars: PIXI.BitmapFont.ASCII.concat(['∀']),
  });

  useEffect(() => {
    console.log("papersData", papersData, "edgesData", edgesData, "clusterData", clusterData)
    const app = new PIXI.Application({
      width: window.innerWidth,
      height: window.innerHeight - 1000,
      resolution: window.devicePixelRatio || 1,
      autoDensity: true,
      backgroundColor: 0x121212,
      resizeTo: window,
    });

    if (pixiContainer.current && pixiContainer.current.childNodes.length > 0) {
      pixiContainer.current.replaceChild(app.view, pixiContainer.current.childNodes[0]);
    } else {
      pixiContainer.current.appendChild(app.view);
    }

    // For now, make the world square and even for better circle parsing
    const viewport = new Viewport({
      screenWidth: window.innerWidth,
      screenHeight: window.innerHeight,
      worldWidth: window.innerWidth,
      worldHeight: window.innerHeight,
      ticker: app.ticker,
      events: app.renderer.events,
      stopPropagation: true,
    });
    viewport.sortableChildren = true;
    viewport.drag().pinch().wheel().decelerate()
      // .clamp({direction: 'all'})
      .clampZoom({ minWidth: 50, maxHeight: viewport.worldHeight, maxWidth: viewport.worldWidth})
      .setZoom(1)
      .moveCenter(0, 0);
    app.stage.addChild(viewport);

    let paperNodes = papersData.map(({title, x, y, citationCount, paperId}) => ({title, x: x, y: y, citationCount, paperId}))
    console.log("PAPER NODES FRESH", paperNodes)

    // Creating voronoi from leaf nodes
    let leafClusters = getLeafClusters(clusterData);

    console.log("LEAFCLUSTERSSSS", leafClusters)
    console.log("paperNodes OG", paperNodes)

    // Creates a map from cluster_id to main_topic
    const clusterMap = new Map();
    function traverseCluster(cluster) {
        clusterMap.set(cluster.cluster_id, cluster.main_topic);

        // Check if this cluster has child clusters.
        if (cluster.content && typeof cluster.content[0] === 'object') {
            cluster.content.forEach(childCluster => traverseCluster(childCluster));
        }
    }
    clusterData.forEach(cluster => traverseCluster(cluster));

    
    // Compute force-directed layout of PaperNodes
    console.log("PRE-LAYOUT PAPERNODES", paperNodes, leafClusters)
    const layout = computeLayout(paperNodes, edgesData, leafClusters);
    paperNodes = layout.paperNodes;
    const centerNodes = layout.centerNodes;
    console.log("POST-LAYOUT PAPERNODES", paperNodes, layout.centerNodes)

    const min_scale = Math.min(...paperNodes.map((node) => Math.sqrt(node.citationCount))) + 1;
    const max_scale = Math.max(...paperNodes.map((node) => Math.sqrt(node.citationCount)));
    console.log("min_scale", min_scale, "max_scale", max_scale)

    // Create dummy 'center' nodes and links to them for each leafCluster
    let centroidNodes = []
    leafClusters.forEach(cluster => {
      // Link all nodes in the cluster to the 'center' node
      let sumX = 0;
      let sumY = 0;
      let count = 0;
    
      cluster.content.forEach(paperId => {
        // Find the paperNode with the same paperId
        let paperNode = paperNodes.find(node => node.paperId === paperId);
        if (paperNode) {
          sumX += paperNode.x;
          sumY += paperNode.y;
          count++;
        }
      });
    
      // Calculate and set the centroid for this cluster
      let centroidX = count > 0 ? sumX / count : undefined;
      let centroidY = count > 0 ? sumY / count : undefined;
    
      cluster.centroid_x = centroidX;
      cluster.centroid_y = centroidY;

      centroidNodes.push({x: centroidX, y: centroidY, cluster_id: cluster.cluster_id, citationCount: cluster.citationCount, main_topic: cluster.main_topic})
    });    
    
    // Calculating voronoi from true centroids of leaf clusters
    const extendFactor = 100 // hardcoding for circle design
    const delaunay = Delaunay.from(centroidNodes.map((node) => [node.x, node.y]));
    const minX = Math.min(...paperNodes.map((paper) => paper.x));
    const maxX = Math.max(...paperNodes.map((paper) => paper.x));
    const minY = Math.min(...paperNodes.map((paper) => paper.y));
    const maxY = Math.max(...paperNodes.map((paper) => paper.y));
    const voronoi = delaunay.voronoi([minX - extendFactor, minY - extendFactor, maxX + extendFactor, maxY + extendFactor]);

    // Adding a circle mask
    let farthestDistance = 0;
    paperNodes.forEach(node => {
      let distance = Math.sqrt(Math.pow(node.x, 2) + Math.pow(node.y, 2));
      if (distance > farthestDistance) {
        farthestDistance = distance;
      }
    });
    let circleMask = new PIXI.Graphics();
    circleMask.beginFill(0x000000); // You can fill with any color
    circleMask.drawCircle(0, 0, farthestDistance + 10);
    circleMask.endFill();
    viewport.mask = circleMask;
    viewport.addChild(circleMask);

    // Ensuring parentIds extend to the farthest zoom
    const zoomLayers = 25
    leafClusters.forEach((node) => {
        let lastParentId = node.cluster_id; // fallback to node's cluster_id
        for (let zoomLevel = 0; zoomLevel < zoomLayers; ++zoomLevel) {
            if (node.parents[zoomLevel] !== undefined) {
                lastParentId = node.parents[zoomLevel];
            } else {
                node.parents[zoomLevel] = lastParentId;
            }
        }
    });
    console.log("leafclusters after adding", leafClusters)

    // Creates a map from parentId to nodes with that parentId
    const clusterGroups = new Map();
    for (let zoomLevel = 0; zoomLevel < zoomLayers; ++zoomLevel) {
        leafClusters.forEach((node, i) => {
            const parentId = node.parents[zoomLevel];
            const key = [parentId, zoomLevel];
            if (!clusterGroups.has(key)) {
                clusterGroups.set(key, []);
            }
            clusterGroups.get(key).push(node);
        });
    }

    // Calculates centroid for each cluster
    const clusterCentroids = new Map();
    for (let [[clusterId, zoomLevel], nodes] of clusterGroups) {
        let sumX = 0, sumY = 0;
        for (let node of nodes) {
            sumX += node.centroid_x;
            sumY += node.centroid_y;
        }
        const key = [clusterId, zoomLevel].toString();
        clusterCentroids.set(key, {x: sumX / nodes.length, y: sumY / nodes.length, layer: zoomLevel});
    }
    console.log("CLUSTER CENTROIDS", clusterCentroids)

    // Hardcoding (zoomLayers) a parent cluster mapping for voronois
    let parentColorMap = new Map();
    for (let zoomLevel = 0; zoomLevel < zoomLayers; ++zoomLevel) {
      // Setting parent cluster colors by cluster_id
      leafClusters.forEach(node => {
        let parentId = node.parents[zoomLevel];
        if (parentId) {
          if (!parentColorMap.has(parentId)) {
              parentColorMap.set(parentId, colorSequence[parentId % 301]);
          }
        }
      });
    }

    // Create and add all circles and text to the viewport
    const drawNodes = (nodes, viewport) => {
      // let zoomLevel = (viewport.scaled - 1) * 100;
      
      let zoomLevel = Math.round((viewport.scaled - 0.95) * 8)

      // Font size
      const bounds = viewport.getVisibleBounds();
      let min_font_size = bounds.width < bounds.height
          ? bounds.width / (17 * 2)
          : bounds.height / (30 * 2);
      const max_font_size = min_font_size * 1.7;

      // Adding cluster polygons to the viewport
      leafClusters.forEach((node, i) => {
          const parentId = node.parents[zoomLevel];
          let fillColor = colorSequence[node.cluster_id % 301]
          if (parentId) {
            fillColor = parentColorMap.get(parentId);
          }

          if (node.region) {
            node.region.clear()
          } 
            const region = voronoi.cellPolygon(i);

            const lineWidth = 2;
            const lineColor = 0x000000; // black
            const polygon = new PIXI.Graphics();

            polygon.beginFill(fillColor, 0.5);
            // polygon.lineStyle(lineWidth, lineColor); // Add this line to draw the border
            polygon.drawPolygon(region.map(([x, y]) => new PIXI.Point(x, y)));
            polygon.endFill();

            node.region = polygon;
            viewport.addChild(polygon);
      });

      // Adding the cluster text to viewport
      // leafClusters.forEach((node, i) => {
      //     const parentId = node.parents[zoomLevel];

      //     // Create text for top category
      //     let topCategory = "Unknown";
      //     if(parentId){
      //       topCategory = clusterMap.get(parentId)
      //     }
      //     else{
      //       topCategory = clusterMap.get(node.cluster_id);
      //     }
      //     topCategory = topCategory.slice(1, -1);

      //     // Check if node.text already exists
      //     if (node.text) {
      //       viewport.removeChild(node.text);
      //     }

      //     // Create new text

      //     node.text = new PIXI.BitmapText(multilineText(topCategory, 15), {
      //       fontFamily: 'Arial',
      //       fontSize: min_font_size / 2,
      //       fontName: "TopicFont",
      //       fill: 0xFFD700,
      //       align: 'left',
      //       visible: true,
      //       zIndex: 10,
      //     });
          
      //     // Position the text at the centroid of the region
      //     node.text.position.set(node.centroid_x, node.centroid_y);

      //     node.text.anchor.set(0.5, 0);

      //     // Add the text to the viewport
      //     viewport.addChild(node.text);
      // });
      
      clusterCentroids.forEach((centroid, key) => {
        let [clusterId, layer] = key.split(',').map(Number); 
        if (centroid.layer === zoomLevel) {
          let topCategory = "Unknown";
          if(clusterMap.has(clusterId)){
              topCategory = clusterMap.get(clusterId);
          }
          topCategory = topCategory.slice(1, -1);

          // Check if node.text already exists
          if (centroid.text) {
            viewport.removeChild(centroid.text);
          }
          // Create new text
          centroid.text = new PIXI.BitmapText(multilineText(topCategory, 15), {
              fontFamily: 'Arial',
              fontSize: min_font_size / 2,
              fontName: "TopicFont",
              fill: 0xFFD700,
              align: 'left',
              visible: true,
              zIndex: 10,
          });

          // Position the text at the centroid of the cluster
          centroid.text.position.set(centroid.x, centroid.y);

          centroid.text.anchor.set(0.5, 0);

          // Add the text to the viewport
          viewport.addChild(centroid.text);
        } else if (centroid.text) {
          viewport.removeChild(centroid.text);
        }
      });

      // Adding paper nodes to viewport by leaf cluster
      leafClusters.forEach(cluster => {
        // const parentId = cluster.parents[cluster.parents.length - 1];
        // if (parentId) {
        //   if (!parentColorMap.has(parentId)) {
        //       parentColorMap.set(parentId, colorSequence[parentId % 301]);
        //   }
        // }

        // paperNodes.forEach((node, i) => {
        
        let contentSet = new Set(cluster.content);
        let leafClusterNodes = paperNodes.filter(node => contentSet.has(node.paperId));

        leafClusterNodes.forEach((node, i) => {  
          // Handling Node text, draw labels
          const debug_factor = 3
          // const lambda = debug_factor
          const lambda = debug_factor * (Math.sqrt(node.citationCount) - min_scale) / (max_scale - min_scale);
          // console.log("lambda", lambda, node, node.citationCount, min_scale, max_scale)
          const fontSize = min_font_size + (max_font_size - min_font_size) * lambda;
          const circleHeight = 1 + 4 * lambda;

          if(!node.circle) {
              node.circle = new PIXI.Graphics();
              // node.circle.beginFill(0xb9f2ff);
              node.circle.beginFill(colorSequence[cluster.cluster_id]);
              node.circle.drawCircle(node.x, node.y, circleHeight);
              node.circle.endFill();
              viewport.addChild(node.circle);
          } else {
              node.circle.visible = true; // make it visible if it already exists
          }

          // get the true centroid of the paper node (or before force directed depending on preprocessing)
          // if(!node.centroid_circle) {
          //   node.centroid_circle = new PIXI.Graphics();
          //   node.centroid_circle.beginFill(0xFDDC5C);
          //   node.centroid_circle.drawCircle(cluster.centroid_x, cluster.centroid_y, 10);
          //   node.centroid_circle.endFill();
          //   viewport.addChild(node.centroid_circle);
          // } else {
          //     node.centroid_circle.visible = true; // make it visible if it already exists
          // }

          // if(!node.text) {
          //     // TODO: multiline, this can be done in preprocessing
          //     let words = node.title.split(' ');
          //     let lines = [];
          //     let currentLine = '';
              
          //     for (let word of words) {
          //         if ((currentLine + ' ' + word).length > 30) {
          //             lines.push(currentLine);
          //             currentLine = word;
          //         } else {
          //             currentLine += ' ' + word;
          //         }
          //     }
          //     lines.push(currentLine);
              
          //     let multilineTitle = lines.join('\n').trim();
            
              
          //     node.text = new PIXI.BitmapText(multilineTitle, {
          //       fontFamily: 'Arial',
          //       fontSize: fontSize,
          //       fontName: "TitleFont",
          //       fill: 0xffffff,
          //       align: 'left',
          //       visible: true,
          //       zIndex: 10,
          //     });
          //     node.text.position.set(node.x + circleHeight, node.y + circleHeight);
          //     node.text.anchor.set(0.5, 0);
          //     viewport.addChild(node.text);
          // } else {
          //     node.text.fontSize = fontSize;
          //     node.text.visible = true; // make it visible if it already exists

          //     // Remove overlap between text, I think getBounds can get approximated if it's too slow
          //     const node_bound = node.text.getBounds();
          //     for (let j = 0; j < i; j++) {
          //         const other = nodes[j];
          //         if (other.text.visible && rectIntersectsRect(node_bound, other.text.getBounds())) {
          //             node.text.visible = false;
          //             break;
          //         }
          //     }
          // }
        });

      })
      
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
    }

    // Update visibility of circles and text based on the current field of view and zoom level
    const updateNodes = () => {
      if (!paperNodes) return;

      // reset all nodes and labels graphics
      for (const node of paperNodes) {
        if (node.circle) { node.circle.visible = false };
        if (node.text) { node.text.visible = false };
			}

      // get the current field of view
      const viewport_bounds = viewport.getVisibleBounds();
			viewport_bounds.pad(viewport_bounds.width * 0.2);
      let vis_nodes = paperNodes.filter((node) =>
				viewport_bounds.contains(node.x, node.y)
			)

      // Take the top 15 visible nodes
      vis_nodes.sort((a, b) => {
				return b.citationCount - a.citationCount;
			});
      // vis_nodes = vis_nodes.slice(0, 20);

      // Update visibility of nodes and labels
      drawNodes(vis_nodes, viewport);
    };

    // Update nodes based on ticker
    app.ticker.add(updateNodes)

  }, [papersData, edgesData, clusterData]);

  return <div className="pixiContainer" style={{ display: "flex" }} ref={pixiContainer} />;
};

export default ResearchPaperPlot;