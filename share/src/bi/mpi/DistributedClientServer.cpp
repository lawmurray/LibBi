/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "DistributedClientServer.hpp"

bi::DistributedClientServer::DistributedClientServer() :
    client(NULL), server(NULL) {
  //
}

void bi::DistributedClientServer::setClient(Client* client) {
  this->client = client;
}

void bi::DistributedClientServer::setServer(Server* server) {
  this->server = server;
}
