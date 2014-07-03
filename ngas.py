# -*- coding: utf-8 -*-
import logging
import random
import math
import time
import sys
from collections import deque
import argparse

import numpy as np
import networkx as nx
import pygame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    parser.add_argument('--tmax', type=int, default=10000)
    parser.add_argument('--delta_c', type=float, default=0.05)
    parser.add_argument('--delta_n', type=float, default=0.0005)
    parser.add_argument('--max_age', type=int, default=25)
    parser.add_argument('--lambda_', type=int, default=100)
    parser.add_argument('--nNodes', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.0005)
    parser.add_argument('--slow', action='store_true', default=False)
    return parser.parse_args()


class GasNode(object):

    def __init__(self, w, e):
        self.w = w
        self.e = e

    def get_distance(self, x):
        return math.sqrt((self.w[0] - x[0])**2 + (self.w[1] - x[1])**2)

    def move_to(self, x, delta):
        displacement = [delta * (x[0] - self.w[0]) , delta * (x[1] - self.w[1])]
        self.w[0] += displacement[0]
        self.w[1] += displacement[1]

    def __repr__(self):
        return '<' + repr(id(self)) + ' w:{}'.format(self.w) + ' e:{}'.format(self.e) +'>'


def gas(dataset, tmax=10000, delta_c=0.05, delta_n=0.0005, max_age=25,
        lambda_=100, nNodes = 100, alpha=0.5, beta=0.0005):
    G = nx.Graph()
    
    w1 = [random.random(), random.random()]
    while True:
        w2 = [random.random(), random.random()]
        if w2 != w1:
            break
    
    v1, v2 = GasNode(w1, 0), GasNode(w2, 0)
    G.add_node(v1)
    G.add_node(v2)
    G.add_edge(v1, v2, age=0)
    
    to_return = []
    for t in xrange(1, tmax):
        random.seed(time.time())
        sample = random.choice(dataset)
        
        nodelist = G.nodes()
        distances = sorted([(i, n.get_distance(sample))
                           for i, n in enumerate(nodelist)],
                           key=lambda e: e[1])
        
        c, sc = nodelist[distances[0][0]], nodelist[distances[1][0]]
        
        # move the winner node vs towards x
        c.e += distances[0][1]
        c.move_to(sample, delta_c)

        #move all topological neighbors vn of vs towards x
        winner_neighbors = G.neighbors(c)
        for n in winner_neighbors:
            n.move_to(sample, delta_n)
            G[c][n]['age'] += 1

        if sc in winner_neighbors:
            G[c][n]['age'] = 0
        else:
            G.add_edge(c, sc, age=0)

        for e in G.edges_iter(data=True):
            if e[2]['age'] > max_age:
                G.remove_edge(e[0], e[1])

        isolated_nodes = [n for n in G.nodes_iter() if G.degree(n) == 0]
        map(lambda n: G.remove_node(n), isolated_nodes)

        nodelist = G.nodes()
        if t % lambda_ == 0 and len(nodelist) <= nNodes:
            maxerror_node = max(nodelist, key=lambda n: n.e)
            maxerror_neighbor = max(G.neighbors(maxerror_node),
                                    key=lambda n: n.e)

            maxerror_node.e *= alpha
            maxerror_neighbor.e *= alpha

            new_w = [0.5 * maxerror_node.w[0] + 0.5 * maxerror_neighbor.w[0],
                     0.5 * maxerror_node.w[1] + 0.5 * maxerror_neighbor.w[1]]
            new_node = GasNode(new_w, maxerror_node.e)
            G.add_node(new_node)

            G.remove_edge(maxerror_node, maxerror_neighbor)
            G.add_edge(maxerror_node, new_node, age=0)
            G.add_edge(new_node, maxerror_neighbor, age=0)

        for n in G.nodes_iter():
            n.e *= beta

        to_return.append((sample, [(e[0].w, e[1].w) for e in G.edges()]))

    return to_return


def main():
    args = parse_args()
    src_nodes = np.loadtxt(args.fname, dtype=int)
    dataset = []
    width, height = src_nodes.shape
    lattice = nx.grid_2d_graph(width, height)
    wall_nodes = map(lambda e: tuple(e),
                     np.transpose(np.nonzero(src_nodes)))
    lattice.remove_nodes_from(wall_nodes)

    for i in xrange(width):
        for j in xrange(height):
            if src_nodes[i, j] != 1:
                dataset.append((i/float(height), j/float(width)))

    def normalize(e):
        e1, e2 = e
        return (int(e1[0]/float(height) * 640), int(e1[1]/float(width) *480)),\
               (int(e2[0]/float(height) * 640), int(e2[1]/float(width) * 480))

    waypoint = map(normalize, lattice.edges())
    graph_data = deque(gas(dataset, tmax=args.tmax, delta_c=args.delta_c,
                          delta_n=args.delta_n, max_age=args.max_age,
        lambda_=args.lambda_, nNodes=args.nNodes, alpha=args.alpha, beta=args.beta))

    # visualization
    screen = pygame.display.set_mode((640,480))

    red = (255,0,0)
    darkBlue = (0,0,128)
    white = (255,255,255)
    black = (0,0,0)

    screen.fill(white)
    pygame.display.update()

    def draw_waypoint(waypoint):
        for p in waypoint:
            for x, y in p:
                pygame.draw.circle(screen, black, (x, y), 3, 1)
            pygame.draw.lines(screen, black, False, p, 1)

    def normalize_coordinates(x, y):
        return (int(x * 640), int(y * 480))

    def normalize_edges(e):
        e1, e2 = e
        x1, y1 = e1
        x2, y2 = e2
        p = (normalize_coordinates(x1, y1), normalize_coordinates(x2, y2))
        return p

    def draw_graph(sample, data):
        # draw sample
        sx, sy = normalize_coordinates(*sample)
        pygame.draw.circle(screen, darkBlue, (sx, sy), 5, 5)
        normdata = map(normalize_edges, data)
        for p in normdata:
            for x, y in p:
                pygame.draw.circle(screen, red, (x, y), 3, 1)
            pygame.draw.lines(screen, red, False, p, 1)

    while (True):
        # do something
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        try:
            sample, data = graph_data.popleft()
            screen.fill(white)
            draw_waypoint(waypoint)
            draw_graph(sample, data)
            # update the screen
            pygame.display.update()
            time.sleep(0.001 if not args.slow else 0.5)
        except IndexError:
            pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
