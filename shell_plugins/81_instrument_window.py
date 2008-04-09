import gtk
import gobject

from gettext import gettext as _

class QTInstruments(gtk.Window):

    def __init__(self):
        gtk.Window.__init__(self, gtk.WINDOW_TOPLEVEL)
        self.move(110, 110)

        self.set_size_request(500, 100)
        self.set_border_width(1)
        self.set_title(_('Instruments'))

        self.connect("delete-event", self._delete_event_cb)

        self._ins_combo = InstrumentDropdown()

        self._ins_textview = gtk.TextView()
        self._ins_text = self._ins_textview.get_buffer()

        self._scrolled_win = gtk.ScrolledWindow()
        self._scrolled_win.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        self._scrolled_win.add(self._ins_textview)

        global instruments
        instruments.connect('instrument-added', self._instruments_changed_cb)
        instruments.connect('instrument-removed', self._instruments_changed_cb)
        instruments.connect('instrument-changed', self._instruments_changed_cb)

        self._vbox = gtk.VBox()
        self._vbox.pack_start(self._ins_combo, False, False)
        self._vbox.pack_start(self._scrolled_win, True, True)

        self.add(self._vbox)

        self._vbox.show_all()

    def _delete_event_cb(self, widget, event, data=None):
        print 'Hiding instruments window, use showinstruments() to get it back'
        self.hide()
        return True

    def _instrument_added_cb(self, sender, instrument):
        self._ins_list.append([instrument])

    def _instrument_removed_cb(self, sender, instrument):
        print 'Instrument removed: %s' % instrument

        i = self._ins_list.get_iter_root()
        while i:
            if self._ins_list.get_value(i, 0) == instrument:
                self._ins_list.remove(i)
                break
            i = self._ins_list.iter_next(i)


    def _instrument_changed_cb(self, sender, instrument, property, value):
        print 'Instrument changed: %s' % instrument

    def _instruments_changed_cb(self, widget, arg=None):
        global instruments
        text = ''
        for (iname, ins) in instruments.get_instruments().iteritems():
            text += 'Instrument: %s\n' % iname
            for (param, popts) in ins.get_parameters().iteritems():
                text += '\t%s: %s\n' % (param, ins.get(param))

        self._ins_text.set_text(text)

def showinstruments():
    global _inswin
    _inswin.show()

def hideinstruments():
    global _inswin
    _inswin.hide()

_inswin = QTInstruments()

def get_inswin():
    global _inswin
    return _inswin

if __name__ == "__main__":
    gtk.main()
