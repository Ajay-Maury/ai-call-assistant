import logging
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client

from aiVoiceAssistant.settings import TWILIO_SID, TWILIO_TOKEN

# You can externalize these if needed
WEB_SOCKET_URL = "wss://3nhvvprj-8765.inc1.devtunnels.ms/ws"
VOICE_ROUTE_URL = "https://3nhvvprj-5001.inc1.devtunnels.ms/voice"

client = Client(TWILIO_SID, TWILIO_TOKEN)

@api_view(['GET'])
def health_check(request):
    return Response({"message": "Welcome to AI voice assistant service"})


@api_view(['POST'])
def voice(request):
    try:
        call_sid = request.data.get("CallSid")
        if not call_sid:
            logging.warning("Missing CallSid in request")
            return Response({"error": "Missing CallSid"}, status=status.HTTP_400_BAD_REQUEST)

        logging.info(f"Received CallSid: {call_sid}")

        response = VoiceResponse()
        connect = Connect()
        connect.stream(url=WEB_SOCKET_URL)
        response.append(connect)
        response.pause(length=60)

        return HttpResponse(str(response), content_type="text/xml")

    except Exception as e:
        logging.exception("Error handling /voice request")
        return Response(
            {"error": "Internal Server Error", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def make_call(request):
    try:
        to_number = request.data.get("to")
        if not to_number:
            logging.warning("Missing 'to' number in request")
            return Response({"error": "Missing 'to' number"}, status=status.HTTP_400_BAD_REQUEST)

        if not TWILIO_NUMBER or not VOICE_ROUTE_URL:
            logging.error("Missing Twilio config")
            return Response({"error": "Server misconfiguration"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        logging.info(f"Initiating call from {TWILIO_NUMBER} to {to_number}")
        call = client.calls.create(
            to=to_number,
            from_=TWILIO_NUMBER,
            url=VOICE_ROUTE_URL,
        )

        return Response({"message": "Call initiated", "sid": call.sid})

    except Exception as e:
        logging.exception("Error initiating call")
        return Response(
            {"error": "Internal Server Error", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
