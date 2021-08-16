
message AirFrameMsg
{
    int type;
    int source;
    int destination;
	// Coordinates information
    double x;
    double y;
    double z;
    double destX;
    double destY;
    double destZ;
    double timeStamp;
    // all physical information
    int channelMode;
    double txPowerUpdate;
    double centerFreq;		// Magnitude : GHz
    // SRS external tx_grid filename
    string txGridName;
}
