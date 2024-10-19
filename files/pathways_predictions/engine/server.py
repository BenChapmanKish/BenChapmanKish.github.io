#!/usr/bin/python3
# ECE 498B 2022
# Ben Chapman-Kish

# Main event loop to receive prediction requests and return results via sockets

from __future__ import annotations
from shared.types import *
from shared.serial import Serializer
from engine.predictor import PredictionEngine

import asyncio
from timeit import default_timer

engine = PredictionEngine()

RESPONSE_BYTES_UNKNOWN_FAILURE = Serializer.serialize(Response(success=False, reason=FailureReason.UNKNOWN))
RESPONSE_BYTES_INVALID_REQUEST = Serializer.serialize(Response(success=False, reason=FailureReason.INVALID_REQUEST))
RESPONSE_BYTES_REQUEST_CANCELLED = Serializer.serialize(Response(success=False, reason=FailureReason.REQUEST_CANCELLED_BY_SERVER))

def fulfill_request(request: Request) -> Response:
    if request.type in (RequestType.PING, RequestType.DISCONNECT):
        return Response(success=True)

    elif request.type == RequestType.POSTINGS_CHUNK:
        if len(request.postings) < 1:
            log.warning("Received no postings to add")
            return Response(success=False, reason=FailureReason.NO_RECORDS_PROVIDED)

        engine.cache_postings(request.postings)
        return Response(success=True)
    
    elif request.type == RequestType.POSTINGS_DONE:
        if request.quantity is None:
            log.warning("Received malformed request object, missing fields")
            return Response(success=False, reason=FailureReason.INVALID_REQUEST)
        
        (new_postings, removed_postings) = engine.get_new_cached_postings_and_reset()

        log.info(f"Cached {new_postings} new postings and removed {removed_postings} expired postings (out of {request.quantity} received in total)")
        log.info("")
        return Response(success=True)

    elif request.type == RequestType.REVIEWS_CHUNK:
        if len(request.reviews) < 1:
            log.warning("Received no job/course reviews")
            return Response(success=False, reason=FailureReason.NO_RECORDS_PROVIDED)
        
        engine.cache_reviews(request.reviews)
        return Response(success=True)
    
    elif request.type == RequestType.REVIEWS_DONE:
        if request.quantity is None:
            log.warning("Received malformed request object, missing fields")
            return Response(success=False, reason=FailureReason.INVALID_REQUEST)
        
        new_reviews = engine.get_new_cached_reviews_and_reset()
        
        log.debug(f"Cached {new_reviews} new reviews (out of {request.quantity} received in total)")
        log.debug("")

        if new_reviews == 0:
            return Response(success=True)

        timing_start = default_timer()

        engine.learn_new_reviews()

        timing_end = default_timer()
        elapsed = timing_end - timing_start

        log.info(f"Learned {new_reviews} new reviews in {elapsed:.3f} seconds")
        log.info("")
        return Response(success=True)

    elif request.type == RequestType.SUGGESTIONS:
        if request.posting_type is None or request.user is None:
            log.warning("Received malformed request object, missing fields")
            return Response(success=False, reason=FailureReason.INVALID_REQUEST)

        if not engine.can_make_suggestions(request.posting_type):
            log.warning(f"Cannot create {request.posting_type.name} suggestions yet as there isn't enough training data")
            return Response(success=False, reason=FailureReason.INSUFFICIENT_TRAINING_DATA)
        
        log.info(f"Creating {request.posting_type.name} suggestions for user {request.user.id}...")
        
        timing_start = default_timer()

        results = engine.make_suggestions_for_user(request.user.features, request.posting_type)

        timing_end = default_timer()
        elapsed = timing_end - timing_start

        if results is None:
            log.warning(f"No {request.posting_type.name} postings exist?? ")
            return Response(success=False, reason=FailureReason.NO_POSTINGS_MATCHING_FILTERS)
        
        elif len(results) < 1:
            log.warning(f"Could not generate any {request.posting_type.name} suggestions for user '{request.user.id}'")
            return Response(success=False, reason=FailureReason.NO_SUGGESTIONS_AVAILABLE)

        log.info(f"Generated {len(results)} {request.posting_type.name} suggestions for user '{request.user.id}' in {elapsed:.3f} seconds")
        log.info("")
        return Response(success=True, suggestions=results)
    
    return Response(success=False, reason=FailureReason.INVALID_REQUEST)

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peername: str = writer.get_extra_info('peername')
    client_addr = f"{peername[0]}:{peername[1]}"
    log.info(f"Connected established with client: {client_addr}")

    request = Request(RequestType.PING)
    response_bytes = RESPONSE_BYTES_UNKNOWN_FAILURE
    
    try:
        while request.type != RequestType.DISCONNECT:
            request_bytes = (await reader.readline())

            if not request_bytes:
                break

            try:
                request: Request = Serializer.deserialize(request_bytes, Request)

                

                if request.type not in (RequestType.PING, RequestType.DISCONNECT, RequestType.POSTINGS_CHUNK, RequestType.REVIEWS_CHUNK):
                    log.debug("")
                    log.debug(f"Received {request} from {client_addr}")
                
                response = fulfill_request(request)
                response_bytes = Serializer.serialize(response)
                
                if request.type not in (RequestType.PING, RequestType.DISCONNECT, RequestType.POSTINGS_CHUNK, RequestType.REVIEWS_CHUNK):
                    log.debug(f"Sending {response} to {client_addr}")
                    log.debug("")

            except (Serializer.DECODE_ERROR, EOFError) as e:
                log.exception(f"Could not deserialize request: {e}")
                response_bytes = RESPONSE_BYTES_INVALID_REQUEST
            
            except KeyboardInterrupt:
                log.warning(f"Received keyboard interrupt while handling request, this will result in unexpected behaviour!")
                response_bytes = RESPONSE_BYTES_REQUEST_CANCELLED
            
            except Exception as e:
                log.exception(f"Unknown exception encountered: {e}")
            
            finally:
                writer.write(response_bytes)
                await writer.drain()
        
        log.info(f"Closing connection with client: {client_addr}")
        writer.close()

    except ConnectionError as e:
        log.info(f"Connection broken with client: {client_addr} {e}")

async def run_server_async() -> None:
    server = await asyncio.start_server(handle_client, Config.engine_host, Config.engine_port)

    sockname = server.sockets[0].getsockname()
    log.info(f"Started predictions server on: {sockname}")
    log.info("")

    async with server:
        await server.serve_forever()

def main() -> None:
    log.info("Initializing predictions server...")

    try:
        asyncio.run(run_server_async(), debug=Config.debug_server)

    except (KeyboardInterrupt, SystemExit):
        log.info("")
        log.info("Received keyboard interrupt, closing server")

if __name__ == "__main__":
    main()
